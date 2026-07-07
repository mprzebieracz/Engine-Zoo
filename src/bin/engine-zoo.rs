#[tokio::main]
async fn main() -> anyhow::Result<()> {
    engine_zoo::app::run().await
}
