"""SQL query string constants for system monitoring and evaluation."""

from __future__ import annotations

# -----------------------------------------------------------------------
# Compression monitoring
# -----------------------------------------------------------------------

COMPRESSION_MONITORING = """\
SELECT date(created_at) as day,
       COUNT(*) as observations,
       AVG(json_extract(data, '$.ratio')) as avg_ratio,
       AVG(duration_ms) as avg_compress_ms,
       SUM(tokens_in) as total_input_tokens,
       SUM(tokens_out) as total_output_tokens,
       ROUND(
           SUM(tokens_in) * 1.0 / 1000000
           + SUM(tokens_out) * 5.0 / 1000000,
           4
       ) as cost_usd
FROM event_log
WHERE event_type = 'compress.done'
GROUP BY day ORDER BY day
"""

COMPRESSION_ERRORS = """\
SELECT date(created_at) as day,
       COUNT(*) as errors,
       json_extract(data, '$.error_type') as error_type
FROM event_log
WHERE event_type = 'compress.error'
GROUP BY day, error_type
ORDER BY day DESC
"""

# -----------------------------------------------------------------------
# Search quality
# -----------------------------------------------------------------------

SEARCH_QUALITY = """\
SELECT date(created_at) as day,
       COUNT(*) as queries,
       AVG(json_extract(data, '$.result_count')) as avg_results,
       AVG(json_extract(data, '$.top_score')) as avg_top_score,
       AVG(duration_ms) as avg_ms
FROM event_log
WHERE event_type LIKE 'search.%'
GROUP BY day ORDER BY day
"""

# -----------------------------------------------------------------------
# AST resolution accuracy
# -----------------------------------------------------------------------

AST_RESOLUTION = """\
SELECT date(created_at) as day,
       AVG(
           1.0 - CAST(json_extract(data, '$.unresolved_calls') AS REAL) /
           MAX(json_extract(data, '$.call_edges'), 1)
       ) as avg_resolution_rate,
       SUM(json_extract(data, '$.call_edges')) as total_edges
FROM event_log
WHERE event_type = 'ast.scan'
GROUP BY day ORDER BY day
"""

# -----------------------------------------------------------------------
# Context injection efficiency
# -----------------------------------------------------------------------

CONTEXT_INJECTION = """\
SELECT date(created_at) as day,
       AVG(json_extract(data, '$.total_tokens')) as avg_injected,
       MAX(json_extract(data, '$.total_tokens')) as max_injected,
       AVG(json_extract(data, '$.budget')) as avg_budget,
       AVG(duration_ms) as avg_ms
FROM event_log
WHERE event_type = 'hook.context_inject'
GROUP BY day ORDER BY day
"""

# -----------------------------------------------------------------------
# Learnings health
# -----------------------------------------------------------------------

LEARNINGS_HEALTH = """\
SELECT category,
       COUNT(*) as count,
       AVG(confidence) as avg_confidence,
       SUM(CASE WHEN confidence >= 0.8 THEN 1 ELSE 0 END) as high_confidence,
       SUM(CASE WHEN confidence < 0.3 THEN 1 ELSE 0 END) as low_confidence
FROM learnings
WHERE is_active = 1
GROUP BY category
"""

# -----------------------------------------------------------------------
# Overall system health
# -----------------------------------------------------------------------

SYSTEM_HEALTH_DAILY = """\
SELECT date(el.created_at) as day,
       (SELECT COUNT(*) FROM sessions
        WHERE date(started_at) = date(el.created_at)) as sessions,
       COUNT(DISTINCT CASE WHEN el.event_type = 'compress.done'
           THEN el.id END) as observations,
       COUNT(DISTINCT CASE WHEN el.event_type = 'compress.error'
           THEN el.id END) as compress_errors,
       COUNT(DISTINCT CASE WHEN el.event_type LIKE 'search.%'
           THEN el.id END) as searches,
       COUNT(DISTINCT CASE WHEN el.event_type = 'hook.context_inject'
           THEN el.id END) as injections,
       ROUND(SUM(CASE WHEN el.event_type = 'compress.done'
           THEN el.tokens_in * 1.0 / 1000000 + el.tokens_out * 5.0 / 1000000
           ELSE 0 END), 4) as compress_cost_usd
FROM event_log el
GROUP BY day ORDER BY day DESC LIMIT 14
"""

CUMULATIVE_COST = """\
SELECT SUM(tokens_in) * 1.0 / 1000000
       + SUM(tokens_out) * 5.0 / 1000000 as total_compress_cost,
       COUNT(*) as total_compressions,
       MIN(created_at) as first_compression,
       MAX(created_at) as last_compression
FROM event_log
WHERE event_type = 'compress.done'
"""
