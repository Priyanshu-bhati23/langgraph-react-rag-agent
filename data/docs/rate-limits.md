# Rate limits

Lumen Pay applies rate limits per API key. Limits are counted in a sliding one-minute window.

## Sandbox tier

Sandbox keys are limited to 60 requests per minute. Sandbox limits cannot be raised, because the sandbox shares infrastructure across all developers.

## Production tier

Production keys are limited to 600 requests per minute by default. Accounts that have processed more than 10,000 live payments can ask support to raise the limit to 2,400 requests per minute.

## Exceeding the limit

A request over the limit receives HTTP 429 with a Retry-After header giving the number of seconds to wait. Clients should back off exponentially starting at one second and stop after five attempts. Requests rejected with 429 are not billed.
