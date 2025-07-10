use polars::prelude::*;

#[macro_export]
macro_rules! pydict {
    ($py:expr; $($key:expr => $value:expr),*) => {{
        let _map = ::pyo3::types::PyDict::new($py);

        $(
            let _ = _map.set_item($key, $value);
        )*

        _map
    }};
}

#[macro_export]
macro_rules! pyimport {
    ($mod_name:expr) => {{
        use pyo3::prelude::*;

        Python::with_gil(|py| PyModule::import(py, $mod_name).map(|m| m.unbind()))
    }};
    ($mod_name:expr, $sym_name:expr) => {{
        use pyo3::prelude::*;

        Python::with_gil(|py| {
            PyModule::import(py, $mod_name)
                .and_then(|m| m.getattr($sym_name))
                .map(|m| m.unbind())
        })
    }};
}

// Row try_extract
#[macro_export]
macro_rules! rtx {
    ($row:expr; $($idx:expr => $type:ty), +) => {
        ($( $row[$idx].try_extract::<$type>(), )+)
    };
}

#[macro_export]
macro_rules! optzip {
    ($a:expr, $b:expr) => {
        if let (Some(a), Some(b)) = ($a, $b) {
            Some((a, b))
        } else {
            None
        }
    };

    ($a:expr, $b:expr, $c:expr) => {
        if let (Some(a), Some(b), Some(c)) = ($a, $b, $c) {
            Some((a, b, c))
        } else {
            None
        }
    };

    ($a:expr, $b:expr, $c:expr, $d:expr) => {
        if let (Some(a), Some(b), Some(c), Some(d)) = ($a, $b, $c, $d) {
            Some((a, b, c, d))
        } else {
            None
        }
    };

    ($a:expr, $b:expr, $c:expr, $d:expr, $e:expr) => {
        if let (Some(a), Some(b), Some(c), Some(d), Some(e)) = ($a, $b, $c, $d, $e) {
            Some((a, b, c, d, e))
        } else {
            None
        }
    };

    ($a:expr, $b:expr, $c:expr, $d:expr, $e:expr, $f:expr) => {
        if let (Some(a), Some(b), Some(c), Some(d), Some(e), Some(f)) = ($a, $b, $c, $d, $e, $f) {
            Some((a, b, c, d, e, f))
        } else {
            None
        }
    };
}

pub fn make_list_series<'a, T>(
    name: &str,
    height: usize,
    width: usize,
    vals: impl IntoIterator<Item = impl AsRef<[T::Native]> + 'a>,
) -> Series
where
    T: PolarsNumericType,
    // I: IntoIterator,
    // I::Item: AsRef<[T::Native]> + 'a,
    // IT: AsRef<[T::Native]> + 'a,
{
    let mut builder =
        ListPrimitiveChunkedBuilder::<T>::new(name.into(), height, width, T::get_static_dtype());
    for row in vals {
        builder.append_slice(row.as_ref());
    }

    builder.finish().into_series()
}
