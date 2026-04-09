"use client";

import { useMemo, useState } from "react";

type Article = {
  headline: string;
  source: string;
  date: string;
  link: string;
  summary: string;
  relevance: string;
};

type ApiResponse = {
  type: "news" | "financial";
  query: string;
  answer: string;
  top_articles: Article[];
};

const chips = [
  "What is driving Nvidia today?",
  "Give me a quick outlook on the S&P 500",
  "Summarize the latest news on Tesla",
  "How are bank stocks reacting to rate cuts?",
];

const statRows = [
  { label: "Scope", value: "Global markets" },
  { label: "Mode", value: "Synthesis + sources" },
  { label: "Latency", value: "Live" },
  { label: "Focus", value: "Macro + equities" },
];

export default function Home() {
  const [query, setQuery] = useState("");
  const [reset, setReset] = useState(false);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [result, setResult] = useState<ApiResponse | null>(null);

  const articleCount = useMemo(
    () => (result?.top_articles?.length ? result.top_articles.length : 0),
    [result]
  );

  const onSubmit = async (event: React.FormEvent<HTMLFormElement>) => {
    event.preventDefault();
    const trimmed = query.trim();
    if (!trimmed || loading) return;

    setLoading(true);
    setError(null);

    try {
      const response = await fetch("/api/news", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ query: trimmed, reset }),
      });

      const data = (await response.json()) as ApiResponse & { error?: string };
      if (!response.ok || data.error) {
        throw new Error(data.error || "Unable to fetch response.");
      }

      setResult(data);
    } catch (err) {
      setResult(null);
      setError(err instanceof Error ? err.message : "Something went wrong.");
    } finally {
      setLoading(false);
    }
  };

  return (
    <main className="relative min-h-screen overflow-hidden">
      <div className="bg-field" />
      <div className="grid-overlay" />

      <div className="relative z-10 mx-auto flex w-[min(1100px,92vw)] flex-col gap-12 py-16">
        <header className="flex flex-col gap-8">
          <div className="flex flex-wrap items-center gap-3">
            <div className="w-fit rounded-full border border-white/15 bg-white/5 px-4 py-2 text-xs font-semibold uppercase tracking-[0.25em] text-emerald-200 reveal">
              Signal-Ready Intelligence
            </div>
            <div className="flex items-center gap-2 rounded-full border border-white/10 bg-white/5 px-4 py-2 text-xs text-slate-300 reveal reveal-delay-1">
              <span className="pulse-dot inline-flex h-2 w-2 rounded-full bg-emerald-300" />
              Live MCP routing
            </div>
          </div>
          <div className="grid gap-6 lg:grid-cols-[1.2fr_0.8fr]">
            <div className="flex flex-col gap-5 reveal reveal-delay-1">
              <h1 className="text-4xl font-semibold leading-tight text-white sm:text-5xl">
                FinQuery Atlas
              </h1>
              <p className="max-w-2xl text-lg text-slate-200/80">
                Ask about markets, companies, or macro themes. We scan live
                signals, synthesize the story, and return a clean executive
                brief with credible sources.
              </p>
              <div className="flex flex-wrap gap-3">
                {chips.map((chip) => (
                  <button
                    key={chip}
                    type="button"
                    onClick={() => setQuery(chip)}
                    className="rounded-full border border-white/15 bg-white/5 px-4 py-2 text-sm text-slate-100/90 transition hover:-translate-y-0.5 hover:border-emerald-300/60"
                  >
                    {chip}
                  </button>
                ))}
              </div>
            </div>
            <div className="glass rounded-3xl p-6 reveal reveal-delay-2">
              <div className="flex items-center justify-between">
                <div className="text-xs uppercase tracking-[0.2em] text-slate-300">
                  System pulse
                </div>
                <div className="flex items-center gap-2 text-xs text-emerald-200">
                  <span className="pulse-dot inline-flex h-2 w-2 rounded-full bg-emerald-300" />
                  Active
                </div>
              </div>
              <div className="mt-6 grid gap-3">
                {statRows.map((row) => (
                  <div
                    key={row.label}
                    className="flex items-center justify-between rounded-2xl border border-white/10 bg-slate-900/60 px-4 py-3"
                  >
                    <span className="text-xs uppercase tracking-[0.2em] text-slate-400">
                      {row.label}
                    </span>
                    <span className="text-sm font-semibold text-white">
                      {row.value}
                    </span>
                  </div>
                ))}
              </div>
            </div>
          </div>
        </header>

        <section className="flex flex-col gap-6">
          <form
            onSubmit={onSubmit}
            className="glass flex flex-col gap-6 rounded-3xl p-7 reveal"
          >
            <div className="flex flex-wrap items-center justify-between gap-4">
              <div>
                <h2 className="text-2xl font-semibold text-white">Ask the agent</h2>
                <p className="text-sm text-slate-300">
                  Describe a market theme or company, then run the query.
                </p>
              </div>
              <label className="flex items-center gap-2 text-xs text-slate-300">
                <input
                  type="checkbox"
                  checked={reset}
                  onChange={(event) => setReset(event.target.checked)}
                />
                New session
              </label>
            </div>

            <div className="rounded-2xl border border-white/15 bg-white/5 p-5">
              <label className="text-xs uppercase tracking-[0.2em] text-slate-300">
                Query
              </label>
              <textarea
                className="mt-3 min-h-[150px] w-full resize-none bg-transparent text-lg text-white outline-none placeholder:text-slate-400/70"
                placeholder="Example: What are the most important market themes this week?"
                value={query}
                onChange={(event) => setQuery(event.target.value)}
              />
              <div className="mt-4 flex flex-wrap items-center justify-between gap-4">
                <span className="text-xs text-slate-400">
                  Shift + Enter for a new line
                </span>
                <button
                  type="submit"
                  disabled={loading}
                  className="accent-gradient glow-ring rounded-full px-6 py-2 text-sm font-semibold text-slate-900 transition hover:-translate-y-0.5 disabled:opacity-60"
                >
                  {loading ? "Working..." : "Run query"}
                </button>
              </div>
            </div>
          </form>

          <div className="glass soft-border flex flex-col gap-5 rounded-3xl p-7 reveal reveal-delay-1">
            <div className="flex items-center justify-between">
              <div className="rounded-full bg-amber-400/20 px-3 py-1 text-xs font-semibold uppercase tracking-[0.2em] text-amber-200">
                Result
              </div>
              <div className="text-xs text-slate-400">
                {result ? new Date().toLocaleString() : "Awaiting query"}
              </div>
            </div>

            {error && (
              <div className="rounded-2xl border border-red-400/40 bg-red-500/10 px-4 py-3 text-sm text-red-200">
                {error}
              </div>
            )}

            {!error && !result && !loading && (
              <div className="rounded-2xl border border-white/10 bg-slate-900/60 p-5 text-sm text-slate-300">
                Run a query to see the synthesized answer and source coverage here.
              </div>
            )}

            {loading && (
              <div className="space-y-3">
                <div className="skeleton h-4 w-32 rounded-full" />
                <div className="skeleton h-6 w-3/4 rounded-xl" />
                <div className="skeleton h-20 w-full rounded-2xl" />
                <div className="skeleton h-4 w-24 rounded-full" />
                <div className="skeleton h-16 w-full rounded-2xl" />
              </div>
            )}

            {result && !loading && (
              <>
                <div className="text-xs uppercase tracking-[0.2em] text-emerald-200">
                  {result.type === "news" ? "News Brief" : "Market Answer"}
                </div>
                <h2 className="text-2xl font-semibold text-white">
                  {result.query}
                </h2>
                <p className="rounded-2xl border border-white/10 bg-slate-900/70 p-5 text-sm leading-relaxed text-slate-200">
                  {result.answer}
                </p>

                <div className="flex items-center justify-between">
                  <h3 className="text-lg font-semibold text-white">Top sources</h3>
                  <span className="text-xs text-slate-400">
                    {articleCount} {articleCount === 1 ? "article" : "articles"}
                  </span>
                </div>

                {articleCount === 0 ? (
                  <div className="rounded-2xl border border-white/10 bg-slate-900/60 p-4 text-xs text-slate-300">
                    No linked sources returned for this query.
                  </div>
                ) : (
                  <div className="grid gap-4">
                    {result.top_articles.map((article) => (
                      <article
                        key={`${article.headline}-${article.source}`}
                        className="rounded-2xl border border-white/10 bg-slate-900/60 p-4 transition hover:-translate-y-0.5 hover:border-emerald-300/50"
                      >
                        <h4 className="text-sm font-semibold text-white">
                          {article.headline}
                        </h4>
                        <div className="mt-2 text-xs text-slate-400">
                          {article.source} · {article.date}
                        </div>
                        <p className="mt-3 text-xs leading-relaxed text-slate-200">
                          {article.summary}
                        </p>
                        {article.relevance && (
                          <p className="mt-2 text-xs text-emerald-200/80">
                            {article.relevance}
                          </p>
                        )}
                        {article.link && article.link !== "N/A" && (
                          <a
                            href={article.link}
                            target="_blank"
                            rel="noreferrer"
                            className="mt-3 inline-flex text-xs font-semibold text-emerald-200 hover:text-emerald-100"
                          >
                            Read source
                          </a>
                        )}
                      </article>
                    ))}
                  </div>
                )}
              </>
            )}
          </div>
        </section>
      </div>
    </main>
  );
}
