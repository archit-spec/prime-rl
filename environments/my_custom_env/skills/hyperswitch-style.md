---
name: hyperswitch-style
description: Condensed style and convention guide for the juspay/hyperswitch Rust monorepo. Sourced from .rustfmt.toml, workspace.lints, docs/CONTRIBUTING.md, docs/architecture.md, add_connector.md, and observed PR patterns. Treat the ABSOLUTES section as non-negotiable (CI-enforced). Treat CONVENTIONS as strong defaults to follow unless the task explicitly says otherwise.
---

# Hyperswitch Rust Style & Conventions

## Repo shape

- Cargo workspace at the root; ~40 member crates under `crates/*`.
- Edition 2021, MSRV Rust 1.85.0.
- Two API versions coexist: `v1` (default) and `v2` (feature-gated). Most files have a v1 form; some have a parallel `*_v2.rs`. **Default to the v1 file unless the task names v2.**
- Key crates and their roles:
  - `router` — main service; payment/refund/payout core flows live in `router/src/core/`.
  - `hyperswitch_interfaces` — shared interface traits, error types, and **canonical constants**.
  - `hyperswitch_domain_models` — domain types (ErrorResponse, RouterData, etc.).
  - `hyperswitch_connectors` — connector implementations under `src/connectors/<name>/`.
  - `pm_auth` — payment method authentication flows.
  - `diesel_models` — DB schema, storage types.
  - `scheduler` — Producer + Consumer for deferred tasks.
  - `common_utils`, `common_enums`, `common_types` — pure utilities shared across crates.
  - `masking` — PII protection wrappers (`Secret<T>`).

## ABSOLUTES (CI fails on violation)

CI runs `cargo +nightly fmt --all --check`, `cargo check --features release`, `cargo check --no-default-features --features release,v2`, and `just clippy_v2` with `RUSTFLAGS="-D warnings"`. **Every workspace.lints warning becomes an error.** Concretely:

### Forbidden language constructs
- **No `unsafe` blocks anywhere.** `unsafe_code = "forbid"` at the workspace level.
- **No `.unwrap()` or `.expect()` in production code.** Use `?`, `ok_or`, `ok_or_else`, or explicit `match`. Allowed only inside `#[cfg(test)]`.
- **No `panic!()`, `unreachable!()`, `todo!()`, `unimplemented!()`, `dbg!()`.** Allowed in tests.
- **No `as` conversions.** Use `From`/`Into`/`TryFrom`/`TryInto`:
  ```rust
  // wrong
  let n: u32 = some_i64 as u32;
  // right
  let n = u32::try_from(some_i64).change_context(errors::ApiErrorResponse::InvalidDataValue { field_name: "n" })?;
  ```
- **No direct indexing (`vec[i]`, `s[..n]`).** Use `.get(i)`, `.first()`, `.split_at`, slicing methods.
- **No `print!`/`println!`/`eprint!`/`eprintln!` in production.** Use the `router_env`/`tracing` logger.
- **No `.cloned()` for `Copy` types.** Use `.copied()`.
- **Use `Self`** inside `impl` blocks rather than repeating the type name (`use_self` lint).
- **No `unused_qualifications`.** Don't write `crate::module::Type` when `Type` is already in scope.
- **No `large_futures` warnings.** Box large async state machines if needed.
- **Document `# Panics`** for any function that can panic.

### Formatting (enforced by `cargo +nightly fmt`)
From `.rustfmt.toml`:
- `unstable_features = true` (nightly-only formatter)
- `group_imports = "StdExternalCrate"` — imports MUST be three groups separated by a blank line:
  1. `std::*`
  2. external crates
  3. `crate::*` / `super::*` / `self::*`
- `imports_granularity = "Crate"` — collapse `use foo::a; use foo::b;` into `use foo::{a, b};`

### Commit message format
`<type>(<scope>): <short summary>` — present tense, lowercase, no trailing period.
- Types: `build | chore | ci | docs | feat | fix | perf | refactor | test`
- Scope is usually the crate name (`router`, `pm_auth`, `hyperswitch_interfaces`, etc.) or one of: `changelog`, `config`, `migrations`, `openapi`, `postman`.
- Body is optional but if present, ≥ 20 chars and explains WHY.

## CONVENTIONS (strong defaults — follow unless task overrides)

### Imports
Match the file you're editing. The canonical block looks like:
```rust
use std::{collections::HashMap, sync::Arc};

use common_utils::{errors::CustomResult, ext_traits::AsyncExt};
use error_stack::{report, ResultExt};
use hyperswitch_domain_models::{
    router_data::{ErrorResponse, RouterData},
    types,
};

use crate::{
    consts,
    core::errors::{self, RouterResult},
    routes::SessionState,
};
```

### Error handling
- Domain errors use `error_stack::Result<T, E>` (re-exported as `CustomResult` in many places).
- Convert between error types with `.change_context(NewError::Variant)?` from `error_stack`.
- Attach context with `.attach_printable("reason")?` or `.attach_printable_lazy(|| format!(...))?`.
- Standard error response (`ErrorResponse`) construction uses these fallback constants:
  - `interfaces_consts::NO_ERROR_CODE` (= `"No error code"`)
  - `interfaces_consts::NO_ERROR_MESSAGE` (= `"No error message"`)
  - `interfaces_consts::REQUEST_TIMEOUT_ERROR_CODE` (= `"TIMEOUT"`)
  - `interfaces_consts::REQUEST_TIMEOUT_ERROR_MESSAGE` (= `"Connector did not respond in specified time"`)
- These constants live in `hyperswitch_interfaces::consts`. **That crate is the single source of truth** — do not redefine these strings anywhere else. If you need them in another crate, import directly from `hyperswitch_interfaces::consts` (commonly aliased as `interfaces_consts`).

### HTTP status code semantics
This is a documented codebase convention from `add_connector.md`:
- **2xx in error path** → genuine connector-reported failure → terminal (mark `RefundStatus::Failure`).
- **4xx** → client error → terminal (definitive failure).
- **5xx (500–511) including 504 timeout** → transient server error → DO NOT mark terminal; let scheduler retry. Set `refund_status: None`.
- **Timeout errors specifically** (`code == REQUEST_TIMEOUT_ERROR_CODE`) → do NOT overwrite the refund record's `refund_error_code` / `refund_error_message` fields, since those should reflect genuine connector-provided errors.

### Connector implementation pattern
- `transformers.rs` — request/response types + `TryFrom` conversions to/from `RouterData`.
- `mod.rs` — implements `ConnectorCommon` and `ConnectorIntegration<...>` traits, wires transformers into Hyperswitch's flow.
- Error routing is automatic by HTTP status:
  - 4xx → `get_error_response`
  - 5xx → `get_5xx_error_response`
  - 2xx → `handle_response`
- When a connector returns a single error message field, populate BOTH `message` and `reason` with the same value. `message` drives smart retries; `reason` drives the dashboard.

### PII handling
- Wrap any field that may contain card numbers, secrets, tokens, or personal data in `masking::Secret<T>`.
- Use `.peek()` to read inside trusted code, `.expose()` only at the boundary, `.into_masked()` when serialising headers.

### Naming
- Types, traits, variants: `PascalCase`.
- Functions, methods, fields, modules: `snake_case`.
- Constants: `SCREAMING_SNAKE_CASE`.
- Acronyms in type names: keep the convention used nearby (e.g. `Url` not `URL`, `Api` not `API` in struct names; `URL` is sometimes seen for short constants).

### Logging
- Use `router_env::logger::{info, warn, error, debug}` (or `tracing::{info, warn, ...}`) — never `println!`.
- Prefer structured fields: `logger::info!(merchant_id = ?id, "fetched account");`.
- Never log fields wrapped in `Secret<_>` without going through masking.

### Database / Diesel
- Schema definitions live in `crates/diesel_models/src/schema.rs` (v1) and `schema_v2.rs` (v2).
- Migrations live in `migrations/`. Adding a column to an existing table requires both `up.sql` and `down.sql`.
- Storage trait split: `diesel_models` defines the types; `storage_impl` implements the trait against Postgres + Redis.

## What NOT to do (common agent failure modes)

- ❌ Create `*.md`, `*.backup`, `*.bak`, `IMPLEMENTATION_SUMMARY.md`, or any summary file. The diff IS the deliverable.
- ❌ Add a new field to a public struct without checking every existing constructor — Rust will surface this at compile time but the change ripples.
- ❌ Add a `pub fn helper(...)` to centralise a 3-line pattern. Prefer the inline pattern unless the helper exists already or is asked for.
- ❌ Add a new dependency to `Cargo.toml` to enable a re-export. Use direct imports from the canonical crate at each call site instead.
- ❌ Touch `*_v2.rs` files when the task describes v1 behavior, or vice versa.
- ❌ Use `as` to silence a type error. Use `TryFrom` and propagate the error.
- ❌ Use string literals like `"TIMEOUT"` when a named constant exists in `hyperswitch_interfaces::consts`.
- ❌ Rewrite whole functions when only a few lines need changing. Minimal diff.

## Minimal diff principle

Match the style of the file you're editing exactly. If the surrounding code uses `match`, use `match`. If it uses an `if let`, use `if let`. If imports are alphabetised, alphabetise yours. **The reviewer should be unable to tell which lines were yours and which were already there**, except for the lines that implement the task.

## Quick checklist before submitting

1. Did I touch only files implied by the task? (Verify with `git diff --stat`.)
2. Does `cargo check --features release` pass for the affected crates?
3. Does `cargo clippy` pass with `-D warnings`?
4. Are imports grouped Std / External / Crate with blank lines between, and granular by crate?
5. No `.unwrap()`, `.expect()`, `panic!`, `as`, `dbg!`, direct indexing, or `println!` in production code?
6. For status-code logic: 5xx → retry (None), 4xx/2xx → terminal failure?
7. For timeout errors: not overwriting connector-provided error fields?
8. Did I import shared constants from `hyperswitch_interfaces::consts` rather than redefining?
9. Is the diff as small as possible while still completing the task?
10. Did I avoid writing summary/explanation files?
