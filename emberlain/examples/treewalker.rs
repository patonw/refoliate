use tokio::task;

use emberlain::SourceWalker;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    tracing_subscriber::fmt().init();
    let local = task::LocalSet::new();

    // Requires local since tree-sitter handles cannot be used multi-threaded
    local.spawn_local(async {
        let langspec = include_str!("../etc/languages.yml");
        let mut src_walk = SourceWalker::default();
        src_walk.load_languages(langspec)?;
        src_walk
            .process_matches("./", async move |n, src| {
                println!("^_- Match {n:?}");
                for cap in n.captures {
                    println!(">.> Capture {cap:?} ~~ {:?}", cap.node.utf8_text(src));
                }
            })
            .await?;
        Ok(()) as anyhow::Result<()>
    });

    local.await;

    Ok(())
}
