# Webhooks

## Delivery and retries

Lumen Pay sends each event to your endpoint as an HTTPS POST. Your endpoint must respond with a 2xx status within 10 seconds. Failed deliveries are retried up to 8 times over 72 hours with exponential backoff.

## Verifying signatures

Every webhook carries a Lumen-Signature header containing an HMAC-SHA256 of the raw request body, keyed with your endpoint secret. Reject any request whose signature does not match, and reject timestamps older than five minutes to prevent replay attacks.

## Event ordering

Events are not guaranteed to arrive in order. Use the created timestamp on the event, not arrival order, when reconciling state.
