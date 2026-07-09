<svelte:head>
  <title>engine-zoo connect4</title>
</svelte:head>

<main class="shell">
  <nav>
    <a href="/">Chess</a>
    <strong>Connect4</strong>
  </nav>
  <section>
    <h1>Connect4</h1>
    <p>The backend still supports Connect4. The first frontend pass is focused on chess.</p>
  </section>
</main>

<style>
  :global(body) {
    margin: 0;
    background: #f4f0e7;
    color: #20231f;
    font-family: Inter, ui-sans-serif, system-ui, sans-serif;
  }

  .shell {
    min-height: 100vh;
    padding: 24px;
  }

  nav {
    display: flex;
    gap: 16px;
    border-bottom: 1px solid #d7cdbc;
    padding-bottom: 18px;
  }

  a {
    color: #37624f;
    font-weight: 800;
  }

  section {
    max-width: 680px;
    margin-top: 28px;
    border: 1px solid #d7cdbc;
    border-radius: 8px;
    background: #fffdfa;
    padding: 22px;
  }
</style>
