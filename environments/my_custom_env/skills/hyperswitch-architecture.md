---
name: hyperswitch-architecture
description: Navigation map for the juspay/hyperswitch Rust monorepo. Read this BEFORE grepping. Covers crate ownership, the v1/v2 cfg-flag split, layered ownership (request build vs consume), the typed-ID catalog, and antipatterns that look right but aren't. Examples drawn from real PRs.
metadata:
  type: skill
  priority: high
---

# Hyperswitch repo navigation

You are editing the `juspay/hyperswitch` Rust monorepo. It is huge (~30 crates, ~hundreds of files in `router/`). **Reading this skill saves ~20 wasted grep turns.** It maps where things live and warns about traps that bit prior agents.

---

## TL;DR — five rules you must apply on every task

1. **`#[cfg(feature = "v1")]` and `#[cfg(feature = "v2")]` are everywhere.** Most domain types and many functions have parallel v1/v2 variants. **If you edit a v1 path, search for the v2 sibling and decide whether it also needs the change.** Missing this is the #1 reason agent patches get rejected.

2. **Typed-ID wrappers already exist for almost every "ID I want to compose."** Before writing `format!("{}_{}", merchant_id, customer_id)` or any stringly-typed key, **grep `crates/common_utils/src/id_type/`, `crates/hyperswitch_domain_models/src/vault.rs`, and `*_id.rs`** for an existing wrapper. The codebase prefers `V1VaultEntityId::new(m, c)` over `format!`. (See § Antipatterns.)

3. **Request structs and request-construction sites are in different files.** A bug in a *request shape* lives in `types/`; a bug in a *callsite* lives in `core/`. Don't edit the callsite when the right fix is changing the struct. (See § Layer map.)

4. **`git log -p $BASE_SHA -- <subsystem>` before grepping.** The most recently merged commit often introduced the exact type/helper you're about to reinvent. Specifically: look at the **immediately-preceding commit** on the base ref.

5. **`GlobalCustomerId` (v2) is globally unique by design.** `CustomerId` (v1) is unique only within a merchant. **Do not "fix" cross-merchant scoping on v2 paths** — there's nothing to fix; `GlobalCustomerId` already carries the scope.

---

## Crate ownership map

| Crate | Owns | Touch when… |
|---|---|---|
| `crates/api_models` | HTTP request/response DTOs at the API boundary | Adding/changing an API field |
| `crates/router` | The HTTP service, controllers, core flow logic | Implementing a flow (payments, payouts, refunds, payment methods) |
| `crates/hyperswitch_domain_models` | Domain entities (vault, payment intent, payment attempt, business profile, merchant account) | Adding/changing a domain type that survives across layers |
| `crates/diesel_models` | DB row structs + queries (diesel ORM) | Changing what is persisted |
| `crates/common_utils` | Typed IDs (`id_type/`), error helpers, crypto, encoding | Adding a new ID type, error helper, or generic utility |
| `crates/common_enums` | Enums shared across crates (Currency, Connector, PaymentMethod) | Adding/changing an enum used in API or DB |
| `crates/hyperswitch_connectors` | Per-connector integrations (Adyen, Stripe, Cybersource, …) | Editing connector-specific request/response mapping |
| `crates/hyperswitch_interfaces` | Traits the connectors implement (`PaymentAuthorize`, `Refund`, etc.) | Adding/changing the connector-facing trait surface |
| `crates/external_services` | AWS KMS, S3, HashiCorp Vault, email | Touching infra integrations |
| `crates/redis_interface` | Redis access | Caching, locks, distributed state |
| `crates/openapi` | OpenAPI route definitions | API surface change that ships in docs |
| `crates/masking` | `Secret<T>` type for PII | Anything sensitive (card numbers, secrets, tokens) |
| `crates/payment_methods` | Newer payment-methods crate (v2 path migration in progress) | v2-only payment method work |
| `crates/cards` | Card-number/expiry validation primitives | Card validation logic |

**Heuristic for which crate to start in:**
- "Where is the HTTP handler?" → `crates/router/src/routes/<flow>.rs`
- "What domain type does it pass around?" → `crates/hyperswitch_domain_models/src/<area>.rs`
- "What's persisted?" → `crates/diesel_models/src/<area>.rs`
- "What's the API DTO?" → `crates/api_models/src/<area>.rs`
- "What's the ID's typed wrapper?" → `crates/common_utils/src/id_type/`
- "What's the connector-side mapping?" → `crates/hyperswitch_connectors/src/connectors/<name>.rs`

---

## The v1 / v2 cfg split — the trap

Hyperswitch is mid-migration between two API surfaces. **Most domain types and many functions appear twice**, gated on `#[cfg(feature = "v1")]` and `#[cfg(feature = "v2")]`. They may have different field types, different ID types (`CustomerId` v1 → `GlobalCustomerId` v2), different signatures, or live in entirely different files.

### Protocol when editing a `#[cfg(feature = "v1")]` block

1. `grep -nP '#\[cfg\(feature = "v2"\)\]' <same_file>` — does a v2 sibling exist *in this file*?
2. If not: `rg -l 'pub (struct|fn|enum) <NAME>' crates/` — does it live elsewhere under v2?
3. Decide: **mirror** (same change to v2), **fork** (v1/v2 differ after fix), or **no-op on v2** (justify it).
4. The change is incomplete until you've consciously made this decision.

### Common v1/v2 cfg-pair sites

| File | Has v1/v2 parallel blocks |
|---|---|
| `crates/router/src/core/payment_methods.rs` | Many functions; v2 paths heavy here |
| `crates/router/src/core/payment_methods/cards.rs` | v1 request-construction lives here |
| `crates/router/src/types/payment_methods.rs` | Request struct definitions for both versions |
| `crates/hyperswitch_domain_models/src/payments/` | Payment intent/attempt domain models |
| `crates/api_models/src/payments.rs` | API DTOs split |
| `crates/common_utils/src/id_type/customer.rs` | `CustomerId` (v1) vs `GlobalCustomerId` (v2) |

---

## Layer ownership — where each kind of fix actually lives

| Fix is about… | Edit here |
|---|---|
| API field name, optional/required, validation | `crates/api_models/src/<area>.rs` |
| Domain-model field, invariants | `crates/hyperswitch_domain_models/src/<area>.rs` |
| **Shape of a request struct (e.g. key field's type)** | `crates/router/src/types/<area>.rs` |
| **Value put into a request field at construction** | `crates/router/src/core/<area>/<flow>.rs` |
| Persistence schema | `crates/diesel_models/src/<area>.rs` + migration in `migrations/` |
| Connector mapping | `crates/hyperswitch_connectors/src/connectors/<name>.rs` |

**Trap from PR #11372:** the agent edited `payment_methods.rs` (v2 consumer) when the fix was at `cards.rs:628` (v1 request-construction callsite) AND `types/payment_methods.rs` (request struct definition). It picked the consumption site, not the construction site. **Construction sites are where bug fixes for "wrong value in field X" belong.**

---

## Typed-ID catalog — check before reinventing

`crates/common_utils/src/id_type/` defines newtype-wrapped IDs. Use these instead of raw strings.

| Wrapper | Lives in | Scope |
|---|---|---|
| `MerchantId` | `id_type/merchant.rs` | Merchant tenant ID |
| `CustomerId` | `id_type/customer.rs` (v1) | Unique within a merchant — **not globally unique** |
| `GlobalCustomerId` | `id_type/global_id/` (v2) | **Globally unique across merchants** by design |
| `PaymentId`, `RefundId`, `PayoutId` | `id_type/payment.rs`, `refund.rs`, etc. | Per-flow IDs |
| `ProfileId`, `OrganizationId`, `TenantId` | `id_type/profile.rs`, etc. | Multi-tenancy hierarchy |
| `V1VaultEntityId` (`#[cfg(feature = "v1")]`) | `hyperswitch_domain_models/src/vault.rs` (`MerchantId + CustomerId`) | v1 vault entity scoping |

Each typed ID provides `new(...)`, `get_string_repr() -> String`, and `Serialize`/`Deserialize` (wire format = the `get_string_repr()` string).

**Heuristic:** if you're about to write `format!("{}_{}", some_id, other_id)`, **stop**. There is almost certainly a typed wrapper.

---

## Antipatterns that look right but aren't

1. **`format!("{}_{}", a, b)` as a "key"** — there's almost always a typed wrapper whose `get_string_repr()` produces the same bytes. Use `V1VaultEntityId::new(m, c)`; make the field type the wrapper, not `String`, so the collision bug becomes a compile error.
2. **Editing the value-flow, not the type** — if you're making `field = transform(x)` at a callsite, ask whether `field`'s *type* should change instead. Type-level fixes are smaller and prevent regression.
3. **"Fixing" a v2 path that wasn't broken** — `GlobalCustomerId` is globally unique; re-introducing `merchant_id_{customer_id}` on v2 is churn.
4. **Editing the consumer when the fix is in the producer** — fix where the broken value is *built*, not where it's *used*.
5. **Edit-test-loop without `cargo check` between edits** — `cargo check -p <crate>` is the cheapest oracle; run it after every non-trivial edit.

---

## When this skill is wrong, trust the code

This skill is a static snapshot. If `grep` shows the function lives somewhere unexpected, **trust the live grep**.
