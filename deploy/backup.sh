#!/usr/bin/env bash
# Back up the transkript::studio Postgres DB + stored audio (Linux server).
# Dumps via the Docker container (no host pg_dump needed) and tars the media
# volume; timestamped with retention. Add to cron for daily runs.
#
# Restore:  docker compose -f deploy/docker-compose.yml exec -T db \
#               psql -U tcuser -d tcdb < db-YYYYmmdd-HHMMSS.sql
set -euo pipefail

here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
compose="$here/docker-compose.yml"
out="${OUT_DIR:-$here/backups}"
keep_days="${KEEP_DAYS:-14}"
db_user="${POSTGRES_USER:-tcuser}"
db_name="${POSTGRES_DB:-tcdb}"
mkdir -p "$out"
stamp="$(date +%Y%m%d-%H%M%S)"

echo "Dumping Postgres -> $out/db-$stamp.sql"
docker compose -f "$compose" exec -T db pg_dump -U "$db_user" -d "$db_name" > "$out/db-$stamp.sql"
[ -s "$out/db-$stamp.sql" ] || { echo "pg_dump produced an empty file — is the db up?" >&2; exit 1; }

# media lives in the named docker volume; copy it out via a throwaway container
echo "Archiving media -> $out/media-$stamp.tar.gz"
docker run --rm -v transkript-studio_media:/media -v "$out":/backup alpine \
  tar czf "/backup/media-$stamp.tar.gz" -C /media . 2>/dev/null || echo "no media volume yet — skipped"

echo "Pruning backups older than ${keep_days}d"
find "$out" -type f -mtime "+$keep_days" -print -delete
echo "Backup complete ($stamp)."
