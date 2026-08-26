#!/usr/bin/env bash

# 用法:
# ./check_tavily.sh keys.txt

FILE="$1"

if [ -z "$FILE" ] || [ ! -f "$FILE" ]; then
  echo "Usage: $0 keys.txt"
  exit 1
fi

while IFS= read -r KEY || [ -n "$KEY" ]; do
  # 去掉首尾空格
  KEY=$(echo "$KEY" | xargs)

  # 跳过空行
  [ -z "$KEY" ] && continue

  MASKED="${KEY:0:10}...${KEY: -6}"

  echo "========================================"
  echo "Key: $MASKED"
  echo "========================================"

  RESPONSE=$(curl -s \
    https://api.tavily.com/usage \
    -H "Authorization: Bearer $KEY")

  if echo "$RESPONSE" | jq . >/dev/null 2>&1; then
    echo "$RESPONSE" | jq .
  else
    echo "$RESPONSE"
  fi

  echo
done < "$FILE"