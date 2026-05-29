---
name: hyperswitch-knowledge
description: Knowledge base overview for hyperswitch payment routing engine
metadata:
  type: skill
---

# Hyperswitch Knowledge Briefing

## Available Connector Documentation

Your knowledge base includes official API docs for these payment processors:

- **adyen** (1 docs, ~12,898,856 tokens)
- **authorize-net** (28 docs, ~431,619 tokens)
- **cybersource** (418 docs, ~5,818,065 tokens)
- **klarna** (67 docs, ~159,537 tokens)
- **mollie** (421 docs, ~2,584,577 tokens)
- **nuvei** (109 docs, ~1,035,587 tokens)
- **stripe** (469 docs, ~4,608,002 tokens)
- **worldpay** (16 docs, ~62,681 tokens)

## Network Rules & Regulations

Card network rules (Visa, Mastercard) and PSD2/SCA regulatory docs:

- **mastercard** (~761,446 tokens)
- **psd2** (regulation, ~70,451 tokens)
- **sca-rts** (regulation, ~18,584 tokens)
- **visa** (~410,432 tokens)

## Using the Knowledge Base

- Use `search_docs` in your MCP tools to grep the knowledge base
- Always search before guessing API field names, error codes, or requirements
- Filter by `source` (connector-docs, network-rules, regulation) and `connector` name
- Look for examples, error handling, and auth patterns in the docs
