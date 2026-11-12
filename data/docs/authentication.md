# Authentication

## API keys

Requests are authenticated with a secret key sent as a Bearer token in the Authorization header. Secret keys start with sk_test_ in the sandbox and sk_live_ in production.

## Key rotation

A key can be rolled from the dashboard. The old key keeps working for 24 hours after rolling so deployments can be updated without downtime.

## Restricted keys

Restricted keys grant access to a chosen set of resources, for example read-only access to payments. Use restricted keys for services that do not need full account access.
