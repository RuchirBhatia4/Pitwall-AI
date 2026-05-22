import React, { useEffect, useMemo, useState } from "react";
import { createRoot } from "react-dom/client";
import {
  Activity,
  Play,
  RefreshCw,
  ShieldCheck,
  SlidersHorizontal,
  TableProperties,
} from "lucide-react";
import {
  getCalibratedDegradation,
  getDashboardSummary,
  getHealth,
  getStorageStatus,
  getStrategyRecommendations,
  runCalibrationPipeline,
  runStrategyPipeline,
} from "./api";
import "./styles.css";

function formatValue(value) {
  if (typeof value === "number") {
    return Number.isInteger(value) ? value.toString() : value.toFixed(3);
  }
  if (typeof value === "boolean") {
    return value ? "Yes" : "No";
  }
  return value ?? "";
}

function uniqueValues(rows, column) {
  return [...new Set(rows.map((row) => row[column]).filter(Boolean))].sort();
}

function Metric({ label, value }) {
  return (
    <div className="metric">
      <span>{label}</span>
      <strong>{value}</strong>
    </div>
  );
}

function ToolbarButton({ busy, children, onClick, title }) {
  return (
    <button className="iconButton" disabled={busy} onClick={onClick} title={title} type="button">
      {busy ? <RefreshCw className="spin" size={17} /> : children}
    </button>
  );
}

function DataTable({ columns, rows, title }) {
  return (
    <section className="panel">
      <div className="panelHeader">
        <div>
          <h2>{title}</h2>
          <p>{rows.length} rows</p>
        </div>
        <TableProperties size={20} />
      </div>
      <div className="tableWrap">
        <table>
          <thead>
            <tr>
              {columns.map((column) => (
                <th key={column}>{column.replaceAll("_", " ")}</th>
              ))}
            </tr>
          </thead>
          <tbody>
            {rows.map((row, rowIndex) => (
              <tr key={`${title}-${rowIndex}`}>
                {columns.map((column) => (
                  <td key={column}>{formatValue(row[column])}</td>
                ))}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </section>
  );
}

function FilterSelect({ label, options, value, onChange }) {
  return (
    <label className="filter">
      <span>{label}</span>
      <select value={value} onChange={(event) => onChange(event.target.value)}>
        <option value="">All</option>
        {options.map((option) => (
          <option key={option} value={option}>
            {option}
          </option>
        ))}
      </select>
    </label>
  );
}

function App() {
  const [health, setHealth] = useState(null);
  const [summary, setSummary] = useState(null);
  const [storage, setStorage] = useState(null);
  const [calibrated, setCalibrated] = useState({ columns: [], rows: [] });
  const [strategies, setStrategies] = useState({ columns: [], rows: [] });
  const [driver, setDriver] = useState("");
  const [team, setTeam] = useState("");
  const [compound, setCompound] = useState("");
  const [strategy, setStrategy] = useState("");
  const [busyAction, setBusyAction] = useState("");
  const [error, setError] = useState("");

  async function refreshData() {
    setError("");
    const [healthResult, storageResult, summaryResult, calibrationResult, strategyResult] = await Promise.all([
      getHealth(),
      getStorageStatus(),
      getDashboardSummary(),
      getCalibratedDegradation(),
      getStrategyRecommendations(),
    ]);
    setHealth(healthResult);
    setStorage(storageResult);
    setSummary(summaryResult);
    setCalibrated(calibrationResult);
    setStrategies(strategyResult);
  }

  useEffect(() => {
    refreshData().catch((err) => setError(err.message));
  }, []);

  async function runPipeline(kind) {
    setBusyAction(kind);
    setError("");
    try {
      if (kind === "calibration") {
        await runCalibrationPipeline();
      } else {
        await runStrategyPipeline();
      }
      await refreshData();
    } catch (err) {
      setError(err.message);
    } finally {
      setBusyAction("");
    }
  }

  const filteredCalibrated = useMemo(() => {
    return calibrated.rows.filter((row) => {
      return (
        (!driver || row.driver === driver) &&
        (!team || row.team === team) &&
        (!compound || row.compound === compound)
      );
    });
  }, [calibrated.rows, compound, driver, team]);

  const filteredStrategies = useMemo(() => {
    return strategies.rows.filter((row) => {
      return (
        (!driver || row.driver === driver) &&
        (!team || row.team === team) &&
        (!strategy || row.strategy === strategy)
      );
    });
  }, [driver, strategies.rows, strategy, team]);

  const driverOptions = uniqueValues([...calibrated.rows, ...strategies.rows], "driver");
  const teamOptions = uniqueValues([...calibrated.rows, ...strategies.rows], "team");
  const compoundOptions = uniqueValues(calibrated.rows, "compound");
  const strategyOptions = uniqueValues(strategies.rows, "strategy");
  const recommended = filteredStrategies.filter((row) => row.is_recommended);

  return (
    <main>
      <header className="topbar">
        <div>
          <p className="eyebrow">Formula 1 strategy intelligence</p>
          <h1>PitWall AI</h1>
        </div>
        <div className="statusPill">
          <ShieldCheck size={18} />
          <span>
            {health?.status === "ok" ? "API online" : "Checking API"} ·{" "}
            {storage?.database_active ? "Live database" : "CSV mode"}
          </span>
        </div>
      </header>

      <section className="commandBand">
        <div className="commandCopy">
          <Activity size={22} />
          <div>
            <h2>Race Strategy Workspace</h2>
            <p>Calibrate degradation, generate strategy rankings, and inspect model outputs.</p>
          </div>
        </div>
        <div className="actions">
          <ToolbarButton
            busy={busyAction === "refresh"}
            onClick={() => {
              setBusyAction("refresh");
              refreshData()
                .catch((err) => setError(err.message))
                .finally(() => setBusyAction(""));
            }}
            title="Refresh dashboard data"
          >
            <RefreshCw size={17} />
          </ToolbarButton>
          <button
            className="primaryAction"
            disabled={Boolean(busyAction)}
            onClick={() => runPipeline("calibration")}
            type="button"
          >
            <Play size={17} />
            Calibrate
          </button>
          <button
            className="primaryAction"
            disabled={Boolean(busyAction)}
            onClick={() => runPipeline("strategy")}
            type="button"
          >
            <Play size={17} />
            Strategies
          </button>
        </div>
      </section>

      {error ? <div className="error">{error}</div> : null}

      <section className="metrics">
        <Metric label="Drivers" value={summary?.drivers ?? "-"} />
        <Metric label="Teams" value={summary?.teams ?? "-"} />
        <Metric label="Calibrated Rows" value={summary?.calibrated_rows ?? "-"} />
        <Metric label="Strategy Rows" value={summary?.strategy_rows ?? "-"} />
        <Metric label="Recommended" value={summary?.recommended_strategies ?? "-"} />
      </section>

      <section className="filters">
        <div className="filterTitle">
          <SlidersHorizontal size={18} />
          <span>Filters</span>
        </div>
        <FilterSelect label="Driver" options={driverOptions} value={driver} onChange={setDriver} />
        <FilterSelect label="Team" options={teamOptions} value={team} onChange={setTeam} />
        <FilterSelect
          label="Compound"
          options={compoundOptions}
          value={compound}
          onChange={setCompound}
        />
        <FilterSelect
          label="Strategy"
          options={strategyOptions}
          value={strategy}
          onChange={setStrategy}
        />
      </section>

      <section className="recommendations">
        <h2>Recommended Strategies</h2>
        <div className="recommendationGrid">
          {recommended.length ? (
            recommended.map((row, index) => (
              <article className="recommendation" key={`${row.driver}-${row.strategy}-${index}`}>
                <span>{row.driver}</span>
                <strong>{row.strategy}</strong>
                <p>{row.recommendation_reason ?? "Best ranked strategy for current inputs."}</p>
              </article>
            ))
          ) : (
            <p className="emptyState">No recommendations match the current filters.</p>
          )}
        </div>
      </section>

      <DataTable
        columns={calibrated.columns}
        rows={filteredCalibrated}
        title="Calibrated Degradation"
      />
      <DataTable columns={strategies.columns} rows={filteredStrategies} title="Strategy Rankings" />
    </main>
  );
}

createRoot(document.getElementById("root")).render(<App />);
