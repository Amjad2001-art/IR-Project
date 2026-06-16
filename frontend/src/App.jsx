import { useEffect, useState } from "react";
import "./App.css";

const API_URL = "http://127.0.0.1:8000";

function App() {
  const [query, setQuery] = useState("weight");
  const [dataset, setDataset] = useState("dataset1");
  const [method, setMethod] = useState("bm25");
  const [topK, setTopK] = useState(5);
  const [k1, setK1] = useState(1.5);
  const [b, setB] = useState(0.75);
  const [alpha, setAlpha] = useState(0.6);
  const [useRefinement, setUseRefinement] = useState(false);
  const [usePersonalization, setUsePersonalization] = useState(false);
  const [useTopicDetection, setUseTopicDetection] = useState(false);

  const [loading, setLoading] = useState(false);
  const [response, setResponse] = useState(null);
  const [error, setError] = useState("");

  const [suggestions, setSuggestions] = useState([]);
  const [suggestionLoading, setSuggestionLoading] = useState(false);
  const [suggestionError, setSuggestionError] = useState("");

  const [searchHistory, setSearchHistory] = useState(() => {
    try {
      const savedHistory = localStorage.getItem("ir_search_history");

      if (savedHistory) {
        return JSON.parse(savedHistory);
      }

      return [];
    } catch {
      return [];
    }
  });

  const methods = [
    { value: "tfidf", label: "TF-IDF" },
    { value: "word2vec", label: "Word2Vec" },
    { value: "bm25", label: "BM25" },
    { value: "inverted_index", label: "Inverted Index" },
    { value: "serial_hybrid", label: "Serial Hybrid" },
    { value: "parallel_hybrid", label: "Parallel Hybrid" },
  ];

  useEffect(() => {
    const trimmedQuery = query.trim();

    if (trimmedQuery.length === 0) {
      setSuggestions(searchHistory.slice(0, 6));
      setSuggestionError("");
      setSuggestionLoading(false);
      return;
    }

    if (trimmedQuery.length < 2) {
      setSuggestions([]);
      setSuggestionError("");
      setSuggestionLoading(false);
      return;
    }

    const timer = setTimeout(async () => {
      setSuggestionLoading(true);
      setSuggestionError("");

      try {
        const res = await fetch(`${API_URL}/suggest-query`, {
          method: "POST",
          headers: {
            "Content-Type": "application/json",
          },
          body: JSON.stringify({
            query: trimmedQuery,
            history: searchHistory,
          }),
        });

        if (!res.ok) {
          throw new Error("suggest-query endpoint failed");
        }

        const data = await res.json();

        const filteredSuggestions = (data.suggestions || []).filter(
          (item) => !searchHistory.includes(item)
        );

        setSuggestions(filteredSuggestions);
      } catch {
        setSuggestions([]);
        setSuggestionError("Suggestions are not available. Check /suggest-query backend endpoint.");
      } finally {
        setSuggestionLoading(false);
      }
    }, 400);

    return () => clearTimeout(timer);
  }, [query, searchHistory]);


  function clearSearchHistory() {
    setSearchHistory([]);
    setSuggestions([]);
    setSuggestionError("");
    localStorage.removeItem("ir_search_history");
  }

  async function handleSearch() {
    setLoading(true);
    setError("");
    setResponse(null);

    const payload = {
      query: query,
      dataset: dataset,
      method: method,
      top_k: Number(topK),
      k1: Number(k1),
      b: Number(b),
      alpha: Number(alpha),
      use_refinement: useRefinement,
      use_personalization: usePersonalization,
      use_topic_detection: useTopicDetection,
      history: searchHistory,
    };

    try {
      const res = await fetch(`${API_URL}/search`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify(payload),
      });

      if (!res.ok) {
        throw new Error("Backend error. Make sure FastAPI is running on port 8000.");
      }

      const data = await res.json();
      setResponse(data);

      const cleanQuery = query.trim();

      if (cleanQuery.length > 0) {
        const updatedHistory = [
          cleanQuery,
          ...searchHistory.filter((item) => item !== cleanQuery),
        ].slice(0, 10);

        setSearchHistory(updatedHistory);
        localStorage.setItem("ir_search_history", JSON.stringify(updatedHistory));
      }
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  }

  function getMethodLabel(value) {
    const item = methods.find((m) => m.value === value);
    return item ? item.label : value;
  }

  return (
    <div className="page">
      <header className="hero">
        <div className="hero-text">
          <p className="badge">Service Oriented Architecture</p>
          <h1>Information Retrieval Search Engine</h1>
          <p className="subtitle">
            A web search engine powered by independent services for preprocessing,
            query refinement, indexing, retrieval, ranking, and model loading.
          </p>

          <div className="proof-pills">
            <span>REST API</span>
            <span>FastAPI Backend</span>
            <span>React Frontend</span>
            <span>Saved Models</span>
          </div>
        </div>

        <div className="architecture-card">
          <div className="node frontend">React Frontend</div>
          <div className="arrow">↓</div>
          <div className="node gateway">FastAPI Gateway</div>

          <div className="services-grid">
            <span>Preprocessing</span>
            <span>Query Refinement</span>
            <span>Retrieval</span>
            <span>Indexing</span>
            <span>Ranking</span>
            <span>Model Loader</span>
            <span>Topic Detection</span>
          </div>
        </div>
      </header>

      <main className="layout">
        <section className="panel search-panel">
          <h2>Search Controls</h2>
          <p className="panel-note">
            Choose the dataset, retrieval method, and parameters, then send the query
            to the backend services.
          </p>

          <label className="field">
            <span>Search Query</span>
            <textarea
              value={query}
              onChange={(e) => setQuery(e.target.value)}
              rows={4}
              placeholder="Enter your query..."
            />
          </label>

          <div className="suggestions-box">
            <div className="suggestions-title">
              <span>
                {query.trim().length === 0
                  ? "Recent Search History"
                  : "Query Suggestions"}
              </span>
              {suggestionLoading && <small>Loading...</small>}
            </div>

            {suggestionError && (
              <p className="suggestions-error">{suggestionError}</p>
            )}

            {!suggestionError && suggestions.length === 0 && (
              <p className="suggestions-empty">
                {query.trim().length === 0
                  ? "Your recent searches will appear here after you perform searches."
                  : "Type a query to get spelling correction, expansion, and history-weighted suggestions."}
              </p>
            )}

            {!suggestionError && suggestions.length > 0 && (
              <div className="suggestions-list">
                {suggestions.map((item, index) => (
                  <button
                    key={`${item}-${index}`}
                    type="button"
                    className="suggestion-chip"
                    onClick={() => setQuery(item)}
                  >
                    {item}
                  </button>
                ))}
              </div>
            )}
          </div>

          <div style={{ display: "flex", justifyContent: "flex-end", marginTop: "10px", marginBottom: "14px" }}>
            <button
              type="button"
              className="suggestion-chip"
              onClick={clearSearchHistory}
              disabled={searchHistory.length === 0}
              title="Remove all saved user search history"
            >
              Clear Search History
            </button>
          </div>

          <div className="grid-two">
            <label className="field">
              <span>Dataset</span>
              <select value={dataset} onChange={(e) => setDataset(e.target.value)}>
                <option value="dataset1">Dataset 1 - Webis Touche</option>
                <option value="dataset2">Dataset 2 - Quora</option>
              </select>
            </label>

            <label className="field">
              <span>Search Method</span>
              <select value={method} onChange={(e) => setMethod(e.target.value)}>
                {methods.map((m) => (
                  <option key={m.value} value={m.value}>
                    {m.label}
                  </option>
                ))}
              </select>
            </label>
          </div>

          <div className="grid-two">
            <label className="field">
              <span>Top K Results</span>
              <input
                type="number"
                min="1"
                max="20"
                value={topK}
                onChange={(e) => setTopK(e.target.value)}
              />
            </label>

            <label className="toggle">
              <input
                type="checkbox"
                checked={useRefinement}
                onChange={(e) => setUseRefinement(e.target.checked)}
              />
              <span>Use Query Refinement</span>
            </label>

            <label className="toggle">
              <input
                type="checkbox"
                checked={usePersonalization}
                onChange={(e) => setUsePersonalization(e.target.checked)}
              />
              <span>Use Personalization</span>
            </label>

            <label className="toggle">
              <input
                type="checkbox"
                checked={useTopicDetection}
                onChange={(e) => setUseTopicDetection(e.target.checked)}
              />
              <span>Use Topic Detection</span>
            </label>
          </div>

          <div className="parameters">
            <h3>Advanced Parameters</h3>

            <div className="slider-row">
              <label>k1: {k1}</label>
              <input
                type="range"
                min="0.5"
                max="3"
                step="0.1"
                value={k1}
                onChange={(e) => setK1(e.target.value)}
              />
            </div>

            <div className="slider-row">
              <label>b: {b}</label>
              <input
                type="range"
                min="0"
                max="1"
                step="0.05"
                value={b}
                onChange={(e) => setB(e.target.value)}
              />
            </div>

            <div className="slider-row">
              <label>alpha: {alpha}</label>
              <input
                type="range"
                min="0"
                max="1"
                step="0.05"
                value={alpha}
                onChange={(e) => setAlpha(e.target.value)}
              />
            </div>
          </div>

          <button className="search-button" onClick={handleSearch} disabled={loading}>
            {loading ? "Searching..." : "Search"}
          </button>

          {error && <div className="error-box">{error}</div>}

          <div className="soa-proof">
            <h3>SOA Proof</h3>
            <ul>
              <li>Frontend sends the request only.</li>
              <li>API Gateway receives the request.</li>
              <li>Backend services execute retrieval and ranking.</li>
              <li>Results return as JSON and are displayed here.</li>
            </ul>
          </div>
        </section>

        <section className="panel results-panel">
          <div className="results-header">
            <div>
              <h2>Ranked Results</h2>
              <p>
                Method: <strong>{getMethodLabel(method)}</strong> | Dataset:{" "}
                <strong>{dataset}</strong>
              </p>
            </div>
          </div>

          {!response && !loading && (
            <div className="empty-state">
              <div className="empty-icon">🔎</div>
              <h3>No search executed yet</h3>
              <p>Enter a query and click Search to display ranked results.</p>
            </div>
          )}

          {loading && (
            <div className="empty-state">
              <div className="loader"></div>
              <h3>Searching...</h3>
              <p>The request is being processed by the backend services.</p>
            </div>
          )}

          {response && (
            <>
              <div className="query-summary">
                <div>
                  <span>Original Query</span>
                  <strong>{response.original_query}</strong>
                </div>

                <div>
                  <span>Used Query</span>
                  <strong>{response.used_query}</strong>
                </div>

                <div>
                  <span>Returned Results</span>
                  <strong>{response.results.length}</strong>
                </div>
              </div>

              {response.refinement_info && (
                <div className="refinement-box">
                  <h3>Query Refinement Details</h3>

                  <p>
                    <strong>Corrected Query:</strong>{" "}
                    {response.refinement_info.corrected_query}
                  </p>

                  <p>
                    <strong>Expanded Query:</strong>{" "}
                    {response.refinement_info.expanded_query}
                  </p>

                  {response.refinement_info.history_suggestions &&
                    response.refinement_info.history_suggestions.length > 0 && (
                      <div>
                        <p>
                          <strong>History-Based Suggestions:</strong>
                        </p>

                        <ul>
                          {response.refinement_info.history_suggestions.map((item, index) => (
                            <li key={`${item.suggested_query}-${index}`}>
                              {item.suggested_query} - similarity score:{" "}
                              {item.similarity_score}
                            </li>
                          ))}
                        </ul>
                      </div>
                    )}
                </div>
              )}

              {(response.personalization_info || usePersonalization) && (
                <div className="refinement-box">
                  <h3>Personalization Details</h3>

                  {response.personalization_info ? (
                    <>
                      <p>
                        <strong>Status:</strong>{" "}
                        {response.personalization_info.personalization_applied
                          ? "Applied"
                          : "Not Applied"}
                      </p>

                      <p>
                        <strong>Reason:</strong>{" "}
                        {response.personalization_info.reason}
                      </p>

                      {response.personalization_info.user_profile_terms && (
                        <div>
                          <p>
                            <strong>User Profile Terms:</strong>
                          </p>

                          <ul>
                            {Object.entries(response.personalization_info.user_profile_terms).map(
                              ([term, score]) => (
                                <li key={term}>
                                  {term}: {Number(score).toFixed(3)}
                                </li>
                              )
                            )}
                          </ul>
                        </div>
                      )}
                    </>
                  ) : (
                    <p>
                      Personalization was requested from the interface, but the backend
                      did not return personalization_info. Restart the backend and make
                      sure /search supports use_personalization.
                    </p>
                  )}
                </div>
              )}

              {(response.topic_info || useTopicDetection) && (
                <div className="refinement-box">
                  <h3>Topic Detection Details</h3>

                  {response.topic_info ? (
                    <>
                      <p>
                        <strong>Status:</strong>{" "}
                        {response.topic_info.topic_detection_applied
                          ? "Applied"
                          : "Not Applied"}
                      </p>

                      <p>
                        <strong>Reason:</strong>{" "}
                        {response.topic_info.reason}
                      </p>

                      <p>
                        <strong>Detected Topic:</strong>{" "}
                        {response.topic_info.detected_topic}
                      </p>

                      <p>
                        <strong>Documents Analyzed:</strong>{" "}
                        {response.topic_info.documents_analyzed}
                      </p>

                      {response.topic_info.top_topic_terms &&
                        response.topic_info.top_topic_terms.length > 0 && (
                          <div>
                            <p>
                              <strong>Top Topic Terms:</strong>
                            </p>

                            <ul>
                              {response.topic_info.top_topic_terms.map((item) => (
                                <li key={item.term}>
                                  {item.term}: {Number(item.score).toFixed(3)}
                                </li>
                              ))}
                            </ul>
                          </div>
                        )}
                    </>
                  ) : (
                    <p>
                      Topic Detection was requested from the interface, but the backend
                      did not return topic_info. Restart the backend and make sure /search
                      supports use_topic_detection.
                    </p>
                  )}
                </div>
              )}

              <div className="results-list">
                {response.results.length === 0 && (
                  <div className="empty-state">
                    <h3>No results found</h3>
                    <p>Try another query, dataset, or method.</p>
                  </div>
                )}

                {response.results.map((item) => (
                  <article className="result-card" key={`${item.rank}-${item.doc_id}`}>
                    <div className="rank">#{item.rank}</div>

                    <div className="result-content">
                      <div className="result-meta">
                        <span>Document ID: {item.doc_id}</span>
                        <span>Score: {Number(item.score).toFixed(4)}</span>
                      </div>

                      {item.original_score !== undefined && (
                        <div className="result-meta">
                          <span>
                            Original Score: {Number(item.original_score).toFixed(4)}
                          </span>
                          <span>
                            Personalization Score:{" "}
                            {Number(item.personalization_score || 0).toFixed(4)}
                          </span>
                        </div>
                      )}

                      {item.matched_profile_terms &&
                        item.matched_profile_terms.length > 0 && (
                          <div className="result-meta">
                            <span>
                              Matched Profile Terms:{" "}
                              {item.matched_profile_terms.join(", ")}
                            </span>
                          </div>
                        )}

                      <p>{item.text}</p>
                    </div>
                  </article>
                ))}
              </div>
            </>
          )}
        </section>
      </main>
    </div>
  );
}

export default App;
