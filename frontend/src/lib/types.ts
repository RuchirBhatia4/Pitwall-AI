export type Compound = "SOFT" | "MEDIUM" | "HARD" | "INTERMEDIATE" | "WET" | string;

export interface Risk {
  expected_delta: number;
  p50: number;
  p90: number;
  cvar90: number;
  win_probability: number;
}

export interface StintPlan {
  compound: Compound;
  start_lap: number;
  end_lap: number;
  laps: number;
  age_start: number;
  new: boolean;
}

export interface Strategy {
  name: string;
  stops: number;
  sequence: Compound[];
  pit_laps: number[];
  total: number;
  delta: number;
  windows: [number, number][];
  stints: StintPlan[];
  risk: Partial<Risk>;
}

export interface ActualPlan {
  sequence: Compound[];
  name: string;
  pit_laps: number[];
  stops: number;
}

export interface LikelyPlan extends Strategy {
  stop_probabilities: Record<string, number>;
  start_probabilities: Record<string, number>;
  trained_on_rounds: number;
}

export interface WearProfile {
  mult: number;
  sd: number;
  prior_mult: number;
  prior_sd: number;
  season_races: number;
  practice_runs: number;
  practice_mult: number | null;
  practice_used?: boolean;
  team: string;
}

export interface Prediction {
  alternatives: Strategy[];
  stop_probabilities?: Record<string, number>;
  wear_profile?: WearProfile | null;
  predicted?: Strategy;
  likely?: LikelyPlan;
  first_stop_interval?: [number, number];
  actual?: ActualPlan;
  hindsight?: Strategy;
  actual_vs_hindsight?: number;
}

export interface Driver {
  driver: string;
  number: string;
  name: string;
  team: string;
  color: string;
  grid: number | null;
  headshot?: string | null;
  position?: number | null;
  status?: string;
  points?: number;
}

export interface CompoundParams {
  offset: number;
  deg: number;
  offset_sd: number;
  deg_sd: number;
  max_stint: number;
  evidence_laps: number;
}

export interface TyreModel {
  compounds: Record<string, CompoundParams>;
  fuel_per_lap: number;
  quad: number;
  team_deg: Record<string, Record<string, number>>;
  noise_sd: number;
  source: string;
}

export interface ActualStint {
  stint: number;
  compound: Compound;
  start_lap: number;
  end_lap: number;
  laps: number;
  age_start: number;
  fresh: boolean;
}

export interface Metrics {
  wet: boolean;
  drivers_evaluated: number;
  modal_stops: number | null;
  stops_accuracy: number | null;
  compound_set_accuracy: number | null;
  start_compound_accuracy: number | null;
  first_stop_mae: number | null;
  first_stop_bias: number | null;
  window_coverage: number | null;
  conformal_coverage: number | null;
  median_time_lost_vs_hindsight: number | null;
  baseline_stops_accuracy?: number | null;
  baseline_compound_set_accuracy?: number | null;
  baseline_first_stop_mae?: number | null;
  likely_stops_accuracy?: number | null;
  likely_compound_set_accuracy?: number | null;
  likely_start_compound_accuracy?: number | null;
  likely_first_stop_mae?: number | null;
}

export interface PracticeRun {
  session: string;
  driver: string;
  team: string;
  compound: Compound;
  laps: number;
  age_start: number;
  age_end: number;
  median_time: number;
  raw_slope: number;
  ages: number[];
  times: number[];
}

export interface Weekend {
  total_laps: number;
  pit_loss: number;
  sc_rate: number;
  vsc_rate: number;
  prior_races: number;
  transfer: { beta: number; sd: number; n: number; source: string };
  offset_transfer: { beta: number; sd: number; n: number; source: string };
  short_run_offsets: Record<string, { offset: number; offset_se: number; pairs: number }>;
  prior: TyreModel;
  practice: { compounds: Record<string, { deg: number; deg_se: number; runs: number; laps: number }> };
  model: TyreModel;
  raw_model: TyreModel;
  behaviour: { soft_mult: number; fuel_wear: number; stop_penalty: number; hard_start: number; calibrated_on: number; loss?: number; loss_uncalibrated?: number };
  stop_penalty: number;
  fuel_wear: number;
  first_stop_q: number | null;
}

export interface RaceFile {
  round: number;
  event: string;
  location: string;
  country: string;
  date: string;
  format: string;
  status: "finished" | "upcoming";
  race_start_utc: string;
  weekend: Weekend;
  baseline: ActualPlan | null;
  practice_runs: PracticeRun[];
  drivers: Driver[];
  predictions: Record<string, Prediction>;
  overview: null | {
    total_laps: number;
    drivers: Driver[];
    stints: Record<string, ActualStint[]>;
    neutralised: { sc: [number, number][]; vsc: [number, number][]; red: [number, number][] };
    pit_loss: number | null;
    rain: boolean;
    track_temp: number | null;
    air_temp: number | null;
  };
  metrics: Metrics | null;
}

export interface SeasonRound {
  round: number;
  event: string;
  location: string;
  country: string;
  date: string;
  format: string;
  available: boolean;
  status: string;
  race_start_utc: string;
  seconds_to_start?: number;
  winner?: { driver: string; name: string; team: string; color: string } | null;
  total_laps?: number;
  neutralised?: { sc: [number, number][]; vsc: [number, number][]; red: [number, number][] };
  metrics?: Metrics;
}

export interface Backtest {
  scope: string;
  driver_races: number;
  stops_accuracy: number;
  compound_set_accuracy: number;
  start_compound_accuracy: number;
  first_stop_mae: number;
  window_coverage: number;
  conformal_coverage: number;
  conformal_target: number;
  baseline: {
    description: string;
    stops_accuracy: number;
    compound_set_accuracy: number;
    start_compound_accuracy: number;
    first_stop_mae: number;
  };
  hybrid?: {
    description: string;
    driver_races: number;
    stops_accuracy: number;
    compound_set_accuracy: number;
    start_compound_accuracy: number;
    first_stop_mae: number;
  };
}

export interface Season {
  year: number;
  rounds: SeasonRound[];
  live_round: number | null;
  next_round: number | null;
  backtest: Backtest | null;
}

export interface CarState {
  driver: string;
  number: string;
  team: string;
  color: string;
  position: number | null;
  gap_to_leader: number | null;
  interval: number | null;
  laps_completed: number;
  compound: Compound;
  tyre_age: number;
  stint: number;
  stops: number;
  compounds_used: Compound[];
  in_pit: boolean;
  retired: boolean;
  last_lap: number | null;
}

export interface RaceState {
  source: string;
  round: number;
  event: string;
  location: string;
  lap: number;
  laps_completed_leader: number;
  total_laps: number | null;
  track_status: string;
  updated_at: string;
  cars: CarState[];
  race_control: { lap: number | null; category: string; message: string; flag: string | null }[];
  weather: { track_temp?: number; air_temp?: number; rainfall?: number };
}

export interface Analysis {
  error?: string;
  driver: string;
  lap: number;
  total_laps: number;
  track_status: string;
  car: CarState;
  call: {
    action: "BOX_NOW" | "BOX_SOON" | "STAY_OUT" | "RETIRED" | "FINISHED" | "NO_PLAN";
    headline: string;
    target_lap?: number | null;
    next_compound?: Compound | null;
    window?: [number, number] | null;
    confidence?: number | null;
    margin_to_next?: number | null;
    box_now_cost?: number | null;
    stop_probabilities?: Record<string, number>;
  };
  plans?: Strategy[];
  model?: {
    pit_loss: number;
    pit_losses_observed: number;
    compounds: Record<string, { deg: number; deg_sd: number; offset: number }>;
    live_evidence: Record<string, { prior: number; observed: number; posterior: number; stints: number }>;
    car_deg: {
      mult: number;
      sd: number;
      prior_mult: number;
      prior_sd: number;
      observed_mult: number | null;
      stints: number;
      laps: number;
      compound?: string;
      observed?: number;
      field?: number;
      posterior?: number;
      current_stint_laps?: number;
    } | null;
    stop_penalty: number;
  };
  rejoin?: {
    position: number;
    stop_cost: number;
    ahead: { driver: string; gap: number; compound: string; tyre_age: number } | null;
    behind: { driver: string; gap: number; compound: string; tyre_age: number } | null;
    traffic: boolean;
  } | null;
  threats?: { type: "undercut_threat" | "undercut_opportunity"; driver: string; gap: number | null; gain: number; live: boolean }[];
  rivals?: { driver: string; position: number | null; compound: string; tyre_age: number; next_stop: number | null; next_compound: string | null; plan: string }[];
  tyre_curve?: { compound: string; ages: number[]; predicted: number[]; band: number; observed: { age: number; delta: number }[] } | null;
  drift?: { z: number; alert: boolean; message: string } | null;
}

export interface LiveStatus {
  connected: boolean;
  source: string | null;
  round: number | null;
  error: string | null;
  last_update_age: number | null;
  shared_feed?: boolean;
  viewers?: number;
  openf1_credentials: boolean;
  f1tv_token: boolean;
}
