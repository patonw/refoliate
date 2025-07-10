use anyhow::Result;
use itertools::izip;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{LazyLock, Mutex};

use ndarray::s;
use polars::prelude::*;
use pyo3::prelude::*;
use pyo3_polars::PyDataFrame;

use colorous::CATEGORY10;
use eframe::egui;
use egui::{CentralPanel, Color32, ScrollArea};
use egui_plot::{Legend, Plot, PlotPoint, PlotResponse, Points};

use embasee::{make_list_series, optzip, pydict, pyimport};

static TSNE: LazyLock<Py<PyAny>> = LazyLock::new(|| pyimport!("sklearn.manifold", "TSNE").unwrap());
static UMAP: LazyLock<Py<PyAny>> = LazyLock::new(|| pyimport!("umap", "UMAP").unwrap());
const PALETTE: [colorous::Color; 10] = CATEGORY10;

// Demonstrate various ways of creating list series.
// What we really want are array series, but the API is non-public
fn _list_series_showcase(df: &DataFrame) -> Result<Series> {
    let x_t = Python::with_gil(|py| {
        let tsne = TSNE.bind(py).call1((2,))?;
        let x_t = tsne.call_method1("fit_transform", (PyDataFrame(df.clone()),))?;

        let x_t: Vec<[f32; 2]> = x_t.extract()?;

        Ok::<_, PyErr>(x_t)
    })?;

    // Building a list series procedurally in a for loop. Drawback: verbosity.
    let mut builder = ListPrimitiveChunkedBuilder::<Float32Type>::new(
        "loop_arr".into(),
        10, //x_t.len(),
        1,
        Float32Type::get_static_dtype(), //DataType::Float32,
    );

    // Float32Type::get_static_dtype();

    for row in &x_t {
        builder.append_slice(row);
    }

    let _ = dbg!(builder.finish().into_series());

    // Wrapping the builder in a helper function
    let tsne_ser = dbg!(make_list_series::<Float32Type>(
        "tsne",
        x_t.len(),
        2,
        x_t.iter().map(|it| &it[..]) // hide the fixed size from the type checker
    ));

    // Building a list series via nested iterators. Drawback: allocates a temporary array
    let _ = dbg!(Series::new(
        "iter_arr".into(),
        x_t.iter()
            .map(|x| x.iter().collect::<Series>())
            .collect::<Vec<_>>(),
    ));

    Ok(tsne_ser)
}

fn main() -> Result<()> {
    let ds = linfa_datasets::iris();

    // let ds = Pca::params(4).whiten(true).fit(&ds).unwrap().transform(ds);

    env_logger::init(); // Log to stderr (if you run with `RUST_LOG=debug`).

    let options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default().with_inner_size([350.0, 200.0]),
        ..Default::default()
    };
    // dbg!(ds.sample_iter().map(|(x, y)| (y, x)).into_group_map());

    // let points: Vec<PlotPoint> = ds
    //     .sample_iter()
    //     .map(|(p, y)| PlotPoint::new(p[0], p[1]))
    //     .collect();

    let df = df!(
        "sepal_length" => ds.records.slice(s![.., 0]).to_vec(),
        "sepal_width" => ds.records.slice(s![.., 1]).to_vec(),
        "petal_length" => ds.records.slice(s![.., 2]).to_vec(),
        "petal_width" => ds.records.slice(s![.., 3]).to_vec(),
        "species" => &ds.targets().iter().map(|x| *x as u32).collect::<Vec<_>>(),
    )
    .unwrap();

    let df_x = df
        .clone()
        .lazy()
        .select([all().exclude(["species"]).cast(DataType::Float32)])
        .collect()
        .unwrap();

    let df_tsne = Arc::new(Mutex::new(df! {
        "umap0" => vec![0f32; df.height()],
        "umap1" => vec![0f32; df.height()],
    }?));

    let df_proj = Python::with_gil(|py| {
        let tsne = TSNE.bind(py).call1((2,))?;
        tsne.call_method(
            "set_output",
            (),
            Some(&pydict! { py; "transform" => "polars"}),
        )?;

        let x_t = tsne.call_method1("fit_transform", (PyDataFrame(df_x.clone()),))?;
        let x_t: PyDataFrame = x_t.extract()?;
        let x_t: DataFrame = x_t.into();

        Ok::<_, PyErr>(x_t)
    })?;

    if let Ok(mut df_tsne) = df_tsne.lock() {
        *df_tsne = df_proj.clone();
    }

    let umap_loading = Arc::new(AtomicBool::new(true));
    let df_umap = Arc::new(Mutex::new(df! {
        "umap0" => vec![0f32; df.height()],
        "umap1" => vec![0f32; df.height()],
    }?));

    // TODO: keep instance around to transform new data
    let umap = Python::with_gil(|py| {
        let umap = UMAP.bind(py).call(
            (),
            Some(&pydict! { py;
                // "n_neighbors" => 5
            }),
        )?;
        Ok::<_, PyErr>(umap.unbind())
    })?;

    let umap = Arc::new(Mutex::new(umap));

    {
        let umap = umap.clone();
        let df_x = df_x.clone();
        let df_umap = df_umap.clone();
        let is_loading = umap_loading.clone();

        std::thread::spawn(move || {
            // UMAP doesn't support polars output. Extracting to 2D vec
            let x_umap = Python::with_gil(|py| {
                let umap = umap.lock().unwrap();
                let umap = umap.bind(py);
                // let umap = UMAP.bind(py).call(
                //     (),
                //     Some(&pydict! { py;
                //         // "n_neighbors" => 5
                //     }),
                // )?;

                let x_u = umap.call_method1("fit_transform", (PyDataFrame(df_x.clone()),))?;
                let x_u: Vec<[f32; 2]> = x_u.extract()?;

                Ok::<_, PyErr>(x_u)
            })
            .unwrap();

            // //Generalized:
            // let (x0, x1): (Vec<&f32>, Vec<&f32>) = x_t
            //     .iter()
            //     .filter_map(|it| it.iter().collect_tuple::<(&f32, &f32)>())
            //     .multiunzip();
            let (umap0, umap1): (Vec<_>, Vec<_>) = x_umap.into_iter().map(|[a, b]| (a, b)).unzip();

            let df_proj = df! {
                "umap0" => umap0,
                "umap1" => umap1,
            }
            .unwrap();
            dbg!(&df_proj);

            if let Ok(mut df_umap) = df_umap.lock() {
                *df_umap = df_proj.clone();
            }
            is_loading.store(false, Ordering::Relaxed);
        });
    }

    // let df = dbg!(df.hstack(df_tsne2.get_columns()))?;
    // let df = dbg!(df.hstack(df_umap2.get_columns()))?;

    let _points = izip!(
        df_proj.select_at_idx(0).unwrap().f32()?,
        df_proj.select_at_idx(1).unwrap().f32()?,
    )
    .filter_map(|(x0, x1)| x0.zip(x1))
    .map(|(x0, x1)| PlotPoint::new(x0, x1))
    .collect::<Vec<_>>();

    eframe::run_native(
        "My egui App with a plot",
        options,
        Box::new(|_cc| {
            Ok(Box::new(MyApp {
                df,
                df_tsne,
                df_umap,
                hover_idx: None,
                umap,
                show_umap: false,
                umap_loading,
            }))
        }),
    )
    .unwrap();

    Ok(())
}

struct MyApp {
    df: DataFrame,
    df_tsne: Arc<Mutex<DataFrame>>,
    df_umap: Arc<Mutex<DataFrame>>,
    hover_idx: Option<usize>,
    umap: Arc<Mutex<Py<PyAny>>>,
    show_umap: bool,
    umap_loading: Arc<AtomicBool>,
}

fn extract_f64(df: &DataFrame, colname: &str) -> Result<Float64Chunked> {
    Ok(df
        .column(colname)?
        .cast(&DataType::Float64)?
        .f64()?
        .to_owned())
}

impl eframe::App for MyApp {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        egui::CentralPanel::default().show(ctx, |ui| {
            let proj = if self.show_umap { "umap" } else { "tsne" };
            egui::SidePanel::left("Left panel")
                .width_range(200.0..=400.0) // Doesn't seem to want to resize beyond min
                .show_inside(ui, |ui| {
                    ScrollArea::vertical().show(ui, |ui| {
                        ui.horizontal(|ui| {
                            ui.label("Projection: ");
                            ui.selectable_value(&mut self.show_umap, false, "TSNE");
                            if self.umap_loading.load(Ordering::Relaxed) {
                                ui.spinner();
                            } else {
                                ui.selectable_value(&mut self.show_umap, true, "UMAP");
                            }
                            // ui.spinner();
                        });

                        ui.label(format!("{:?}", self.hover_idx));

                        if let Some(idx) = self.hover_idx
                            && let Some(row) = self.df.get(idx)
                        {
                            // TODO: column names from dataframe
                            ui.label(format!("Sepal Length: {}", row[0]));
                            ui.label(format!("Sepal Width: {}", row[1]));
                            ui.label(format!("Petal Length: {}", row[2]));
                            ui.label(format!("Petal Width: {}", row[3]));
                        }
                    });
                });

            let PlotResponse {
                hovered_plot_item, ..
            } = Plot::new("My Plot")
                // .height(500.0)
                .legend(Legend::default())
                .show(ui, |plot_ui| {
                    // for i in 0..self.df.height() {
                    //     let row = &self.df.get(i).unwrap();
                    //     if let (Ok(x0), Ok(x1)) = rtx!(row; 0 => f64, 1 => f64) {
                    //         let id = format!("point_{i}");
                    //         plot_ui.points(Points::new("scatter", vec![[x0, x1]]).id(id));
                    //     }
                    // }

                    let proj = if self.show_umap { "umap" } else { "tsne" };
                    let proj_df = if self.show_umap {
                        &self.df_umap
                    } else {
                        &self.df_tsne
                    }
                    .lock()
                    .unwrap();

                    let x0 = extract_f64(&proj_df, format!("{proj}0").as_str()).unwrap();
                    let x1 = extract_f64(&proj_df, format!("{proj}1").as_str()).unwrap();
                    let y = self.df.column("species").unwrap().u32().unwrap();

                    izip!(x0.iter(), x1.iter(), y.iter())
                        .filter_map(|(x0, x1, y)| optzip!(x0, x1, y))
                        .enumerate()
                        .for_each(|(i, (x0, x1, y))| {
                            let id = format!("point_{i}");
                            // TODO: find library for categorical, divergent, etc color palettes

                            let color = PALETTE[y as usize % PALETTE.len()];
                            let color = Color32::from_rgb(color.r, color.g, color.b);
                            let name = format!("species {y}");
                            plot_ui.points(Points::new(name, vec![[x0, x1]]).id(id).color(color));
                        });
                    // plot_ui.points(Points::new("scatter", points).id("asdf").name("gfsdqwr"));
                    // .line(Line::new("curve", PlotPoints::Borrowed(&self.points)).name("curve"));
                });

            for i in 0..self.df.height() {
                let id = format!("point_{i}");
                if Some(egui::Id::new(id)) == hovered_plot_item {
                    self.hover_idx = Some(i)
                }
            }
        });
    }
}
