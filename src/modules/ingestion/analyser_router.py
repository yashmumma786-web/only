"""
Pre-Import Analyser Router
"""
import io
import csv
import json
import random as _random
from fastapi import APIRouter, UploadFile, File
from fastapi.responses import JSONResponse, StreamingResponse
from collections import defaultdict
from src.modules.ingestion import services

router = APIRouter()


# Pre-analyser tunables
BLOCK_IMAGE_FLOOR = 5
CSID_IMAGE_CAP = 200
RANDOM_SEED = 42
MULTI_BLOCK_WARNING = 10

REQUIRED_PRE_ANALYSE_COLS = [
    "packing_list_id",
    "company_stone_id",
    "url",
    "block_no",
    "name",  # stone name (second `name` column)
    "vendor_name",  # synthetic — first `name` column in raw header
]

def _parse_csv_with_vendor_quirk(text: str):
    raw_reader = csv.reader(io.StringIO(text))
    try:
        raw_headers = next(raw_reader)
    except StopIteration:
        return [], [], None, None

    name_positions = [i for i, h in enumerate(raw_headers) if h == "name"]
    vendor_idx = name_positions[0] if len(name_positions) >= 2 else None
    name_idx = name_positions[-1] if name_positions else None

    dict_reader = csv.DictReader(io.StringIO(text))
    dict_rows = list(dict_reader)

    raw_rows_iter = csv.reader(io.StringIO(text))
    next(raw_rows_iter, None)  # skip header

    enriched = []
    for d_row, r_row in zip(dict_rows, raw_rows_iter):
        if vendor_idx is not None and vendor_idx < len(r_row):
            d_row["vendor_name"] = r_row[vendor_idx]
        else:
            d_row["vendor_name"] = ""
        enriched.append(d_row)

    return raw_headers, enriched, vendor_idx, name_idx


def _column_check(raw_headers):
    present = set(raw_headers)
    name_count = sum(1 for h in raw_headers if h == "name")
    checks = []
    for col in REQUIRED_PRE_ANALYSE_COLS:
        if col == "vendor_name":
            checks.append(
                {
                    "column": "name (first occurrence — vendor)",
                    "present": name_count >= 2,
                }
            )
        else:
            checks.append({"column": col, "present": col in present})
    return checks, name_count


def _group_by_csid(rows):

    groups = defaultdict(list)
    for row in rows:
        csid = (row.get("company_stone_id") or "").strip()
        if csid:
            groups[csid].append(row)
    return groups


def _build_analysis(raw_headers, rows, foundation_done: set, foundation_all: set):
    """Build the full pre-import analysis dict.

    foundation_done and foundation_all are injected by the caller (orchestrator)
    so this function never opens foundation.db directly.
    """
    column_checks, name_header_count = _column_check(raw_headers)

    csid_groups = _group_by_csid(rows)
    csv_csids = set(csid_groups.keys())

    suppliers = set()
    for row in rows:
        v = (row.get("vendor_name") or "").strip()
        if v:
            suppliers.add(v)

    active_csids, active_dataset_id = services.get_active_dataset_csids()

    genuinely_new = csv_csids - active_csids
    carried_over = csv_csids & active_csids
    will_be_removed = active_csids - csv_csids

    carried_done = carried_over & foundation_done
    carried_missing_embeddings = carried_over - foundation_done

    multi_block = []
    volume_flags = []
    for csid, rows_in_csid in csid_groups.items():
        blocks = {(r.get("block_no") or "").strip() for r in rows_in_csid}
        blocks.discard("")
        block_count = len(blocks)
        sample = rows_in_csid[0]
        supplier = (sample.get("vendor_name") or "").strip()
        stone_name = (sample.get("name") or "").strip()
        if block_count > MULTI_BLOCK_WARNING:
            multi_block.append(
                {
                    "csid": csid,
                    "supplier": supplier,
                    "stone_name": stone_name,
                    "block_count": block_count,
                }
            )
        if len(rows_in_csid) > CSID_IMAGE_CAP:
            volume_flags.append(
                {
                    "csid": csid,
                    "supplier": supplier,
                    "stone_name": stone_name,
                    "image_count": len(rows_in_csid),
                    "block_count": block_count,
                }
            )

    multi_block.sort(key=lambda x: -x["block_count"])
    volume_flags.sort(key=lambda x: -x["image_count"])

    return {
        "column_check": column_checks,
        "name_header_count": name_header_count,
        "counts": {
            "total_image_rows": len(rows),
            "unique_csids": len(csv_csids),
            "unique_suppliers": len(suppliers),
        },
        "overlap": {
            "active_dataset_id": active_dataset_id,
            "active_dataset_csid_count": len(active_csids),
            "foundation_csid_count": len(foundation_all),
            "genuinely_new": len(genuinely_new),
            "carried_over": len(carried_over),
            "will_be_removed": len(will_be_removed),
            "carried_done": len(carried_done),
            "carried_missing_embeddings": len(carried_missing_embeddings),
        },
        "multi_block": multi_block,
        "volume_flags": volume_flags,
        "constants": {
            "BLOCK_IMAGE_FLOOR": BLOCK_IMAGE_FLOOR,
            "CSID_IMAGE_CAP": CSID_IMAGE_CAP,
            "RANDOM_SEED": RANDOM_SEED,
            "MULTI_BLOCK_WARNING": MULTI_BLOCK_WARNING,
        },
    }


def _trim_rows(csid_groups):

    rng = _random.Random(RANDOM_SEED)

    kept_ids = set()
    trimmed_count = 0

    for csid, rows_in_csid in csid_groups.items():
        if len(rows_in_csid) <= CSID_IMAGE_CAP:
            for r in rows_in_csid:
                kept_ids.add(id(r))
            continue

        trimmed_count += 1

        by_block = defaultdict(list)
        for r in rows_in_csid:
            by_block[(r.get("block_no") or "").strip()].append(r)

        floor_picks = []
        leftover = []
        for _block, brows in by_block.items():
            if len(brows) <= BLOCK_IMAGE_FLOOR:
                floor_picks.extend(brows)
            else:
                floor_picks.extend(brows[:BLOCK_IMAGE_FLOOR])
                leftover.extend(brows[BLOCK_IMAGE_FLOOR:])

        if len(floor_picks) >= CSID_IMAGE_CAP:
            for r in floor_picks[:CSID_IMAGE_CAP]:
                kept_ids.add(id(r))
            continue

        remaining = CSID_IMAGE_CAP - len(floor_picks)
        if leftover and remaining > 0:
            sample = rng.sample(leftover, min(remaining, len(leftover)))
        else:
            sample = []
        for r in floor_picks + sample:
            kept_ids.add(id(r))

    return kept_ids, trimmed_count



@router.post("/admin/clean-import/pre-analyse/trimmed-csv")
async def clean_import_pre_analyse_trimmed_csv(file: UploadFile = File(...)):
    try:
        content = await file.read()
        text = content.decode("utf-8-sig")
        raw_headers, rows, _vidx, _nidx = _parse_csv_with_vendor_quirk(text)

        checks, _ = _column_check(raw_headers)
        missing = [c["column"] for c in checks if not c["present"]]
        if missing:
            return JSONResponse(
                {
                    "success": False,
                    "error": "Missing required columns: " + ", ".join(missing),
                },
                status_code=400,
            )

        csid_groups = _group_by_csid(rows)
        kept_ids, trimmed_count = _trim_rows(csid_groups)

        raw_iter = csv.reader(io.StringIO(text))
        next(raw_iter, None)
        all_raw_rows = list(raw_iter)
        
        kept_raw_rows = []
        for d_row, r_row in zip(rows, all_raw_rows):
            if id(d_row) in kept_ids:
                kept_raw_rows.append(r_row)

        out_buf = io.StringIO()
        writer = csv.writer(out_buf)
        writer.writerow(raw_headers)
        writer.writerows(kept_raw_rows)
        body = out_buf.getvalue()

        summary = {
            "original_row_count": len(rows),
            "trimmed_row_count": len(kept_raw_rows),
            "csids_trimmed": trimmed_count,
            "unique_csids": len(csid_groups),
        }
        in_name = file.filename or "upload.csv"
        if in_name.lower().endswith(".csv"):
            out_name = in_name[:-4] + ".trimmed.csv"
        else:
            out_name = in_name + ".trimmed.csv"

        return StreamingResponse(
            iter([body]),
            media_type="text/csv",
            headers={
                "Content-Disposition": f'attachment; filename="{out_name}"',
                "X-Trim-Summary": json.dumps(summary),
                "Access-Control-Expose-Headers": "X-Trim-Summary",
            },
        )
    except Exception as e:
        return JSONResponse({"success": False, "error": str(e)}, status_code=400)
