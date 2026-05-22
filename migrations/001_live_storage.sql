create table if not exists pitwall_table_rows (
    id bigserial primary key,
    dataset text not null,
    row_index integer not null,
    row_data jsonb not null,
    source_path text,
    created_at timestamptz not null default now()
);

create index if not exists idx_pitwall_table_rows_dataset_created
on pitwall_table_rows (dataset, created_at desc);
