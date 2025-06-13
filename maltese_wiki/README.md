# Maltese Wikipedia Extraction

## Extract Wikipedia Revision History

Extract error-correction pairs from Maltese Wikipedia revision history:

```bash
docker run -it --rm --gpus all \
  -v ~/malteseGEC:/app/malteseGEC \
  -v ~/malteseGEC:/home/fed/malteseGEC \
  -w /app/malteseGEC/maltese_wiki \
  maltesegec-ngc-fixed \
  bash -c 'python3 extract_wiki_revisions.py \
    --input /path/to/mtwiki-latest-pages-meta-history.xml \
    --output ../data/authentic/wikipedia/'
```

## Extract from TFRecords

If Wikipedia data is already in TFRecord format:

```bash
docker run -it --rm --gpus all \
  -v ~/malteseGEC:/app/malteseGEC \
  -v ~/malteseGEC:/home/fed/malteseGEC \
  -w /app/malteseGEC/data \
  maltesegec-ngc-fixed \
  bash -c 'python3 extract_wiki_from_tfrecords.py'
```

## Output

Extracted data will be saved to:
- `/data/authentic/wikipedia/wiki_src.txt`
- `/data/authentic/wikipedia/wiki_trg.txt`