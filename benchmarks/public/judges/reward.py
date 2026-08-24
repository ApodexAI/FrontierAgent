"""Official OfficeQA reward function — vendored verbatim from."""

import re


def normalize_text(text: str) -> str:
    if not text:
        raise ValueError("Cannot normalize empty or None text")
    normalized = text.replace('−', '-')
    return re.sub(r'\s+', ' ', normalized).strip()


_CURRENCY_SYMBOLS = r"$£€¥₹¢₩₽"
_NUMBER_BODY = r"\d{1,3}(?:,\d{3})*(?:\.\d+)?|\d+(?:\.\d+)?"


def _normalize_numeric_formatting(text: str) -> str:
    def _accounting_repl(match: re.Match) -> str:
        currency = match.group(1)
        number = match.group(2)
        if currency is None and is_likely_year(float(number.replace(",", ""))):
            return match.group(0)
        return f"-{number}"

    text = re.sub(
        rf"\(\s*([{_CURRENCY_SYMBOLS}])?\s*({_NUMBER_BODY})\s*\)",
        _accounting_repl,
        text,
    )
    return re.sub(rf"[{_CURRENCY_SYMBOLS}]", "", text)


def extract_numbers_with_context(text: str) -> list[tuple[float, str, bool, bool]]:
    if not text:
        raise ValueError("Cannot extract numbers from empty text")

    text = normalize_text(text)
    text = _normalize_numeric_formatting(text)

    text_no_commas = re.sub(
        r'\d{1,3}(?:,\d{3})+(?:\.\d+)?',
        lambda m: m.group().replace(',', ''),
        text,
    )

    numbers_with_context = []
    pattern = r'-?\d+\.?\d*%?'

    for match in re.finditer(pattern, text_no_commas):
        matched_text = match.group()
        if not matched_text or matched_text == '-':
            continue

        has_percent = matched_text.endswith('%')
        num_text = matched_text.rstrip('%')
        is_negative = num_text.startswith('-')

        try:
            num = float(num_text)
        except ValueError as e:
            raise ValueError(f"Failed to parse number from '{matched_text}': {e}") from e

        start = max(0, match.start() - 20)
        end = min(len(text_no_commas), match.end() + 20)
        context = text_no_commas[start:end].lower()

        numbers_with_context.append((num, context, has_percent, is_negative))

    return numbers_with_context


def detect_unit_in_context(context: str) -> tuple[str | None, float]:
    context_lower = context.lower()
    if re.search(r'\btrillions?\b', context_lower):
        return ('trillion', 1e12)
    if re.search(r'\bbillions?\b', context_lower) or re.search(r'\bb\b', context_lower):
        return ('billion', 1e9)
    if re.search(r'\bmillions?\b', context_lower) or re.search(r'\bm\b', context_lower):
        return ('million', 1e6)
    if re.search(r'\bthousands?\b', context_lower) or re.search(r'\bk\b', context_lower):
        return ('thousand', 1e3)
    return (None, 1.0)


def normalize_number_with_units(number: float, context: str) -> tuple[float, str | None]:
    try:
        unit_name, _ = detect_unit_in_context(context)
        return (number, unit_name)
    except Exception as e:
        raise ValueError(f"Failed to normalize number {number} with context '{context}': {e}") from e


def units_compatible(gt_unit: str | None, pred_unit: str | None) -> bool:
    return gt_unit is None or pred_unit is None or gt_unit == pred_unit


def is_likely_year(num: float) -> bool:
    return 1900 <= num <= 2100 and num == int(num)


def has_significant_text(text: str) -> tuple[bool, str]:
    if not text:
        return False, ""

    cleaned = normalize_text(text).lower()
    cleaned = re.sub(r'-?\d+\.?\d*%?', '', cleaned)
    cleaned = re.sub(r'[,]', '', cleaned)

    unit_words = [
        'trillion', 'trillions', 'billion', 'billions', 'million', 'millions',
        'thousand', 'thousands', 'hundred', 'hundreds',
        'percent', 'percentage', '%'
    ]
    for unit in unit_words:
        cleaned = re.sub(r'\b' + unit + r'\b', '', cleaned)

    cleaned = re.sub(r'[^\w\s]', ' ', cleaned)
    cleaned = re.sub(r'\s+', ' ', cleaned).strip()

    has_text = len(cleaned) >= 2
    return has_text, cleaned


def check_text_overlap(gt_text: str, pred_text: str) -> tuple[bool, str]:
    if not gt_text or not pred_text:
        return False, "Empty text in comparison"

    gt_has_text, gt_cleaned = has_significant_text(gt_text)
    pred_has_text, pred_cleaned = has_significant_text(pred_text)

    if not gt_has_text:
        return True, "GT is purely numeric, text check not required"

    if not pred_has_text:
        return False, f"GT has text '{gt_cleaned}' but prediction is purely numeric"

    if gt_cleaned in pred_cleaned:
        return True, f"Text overlap: '{gt_cleaned}' found in prediction"

    if pred_cleaned in gt_cleaned:
        return True, f"Text overlap: prediction text '{pred_cleaned}' matches GT"

    return False, f"Text mismatch: GT='{gt_cleaned}', Pred='{pred_cleaned}'"


def extract_final_answer_from_xml(text: str) -> tuple[str, str | None]:
    if not text:
        return "", None

    matches = list(
        re.finditer(
            r'<FINAL_ANSWER>\s*(.*?)\s*</FINAL_ANSWER>',
            text,
            re.DOTALL | re.IGNORECASE,
        )
    )

    if matches:
        final_answer_match = matches[-1]
        final_answer = final_answer_match.group(1).strip()
        reasoning_before = text[: final_answer_match.start()].strip()
        return final_answer, reasoning_before if reasoning_before else None

    return text, None


def extract_final_answer(text: str) -> str:
    final_answer, _ = extract_final_answer_from_xml(text)
    return final_answer


def fuzzy_match_answer(ground_truth: str, predicted: str, tolerance: float = 0.00) -> tuple[bool, str]:
    if not ground_truth:
        raise ValueError("Ground truth cannot be empty")
    if not predicted:
        return False, "Predicted answer is empty - marked as incorrect"
    if not 0 <= tolerance <= 1:
        raise ValueError(f"Tolerance must be between 0 and 1, got {tolerance}")

    if "unable to determine" in predicted.lower():
        return False, "Answer contains 'Unable to determine' - marked as incorrect"

    try:
        gt_numbers_with_context = extract_numbers_with_context(ground_truth)
        pred_numbers_with_context = extract_numbers_with_context(predicted)
    except Exception as e:
        raise ValueError(f"Failed to extract numbers: {e}") from e

    gt_numbers = [(num, ctx) for num, ctx, _, _ in gt_numbers_with_context]
    pred_numbers = [(num, ctx) for num, ctx, _, _ in pred_numbers_with_context]

    if gt_numbers and pred_numbers:
        if len(gt_numbers) > 1:
            pred_non_years = [(n, c) for n, c in pred_numbers
                             if not is_likely_year(n) or any(is_likely_year(g) for g, _ in gt_numbers)]

            matched_gt = []
            unmatched_gt = []

            for gt_val, gt_context in gt_numbers:
                try:
                    gt_base, gt_unit = normalize_number_with_units(gt_val, gt_context)
                except Exception as e:
                    raise ValueError(f"Failed to normalize GT number {gt_val}: {e}") from e

                found_match = False
                for pred_val, pred_context in pred_non_years:
                    try:
                        pred_base, pred_unit = normalize_number_with_units(pred_val, pred_context)
                    except Exception as e:
                        raise ValueError(f"Failed to normalize prediction number {pred_val}: {e}") from e

                    if not units_compatible(gt_unit, pred_unit):
                        continue

                    if gt_base == 0:
                        if pred_base == 0:
                            text_matches, _ = check_text_overlap(ground_truth, predicted)
                            if text_matches:
                                found_match = True
                                break
                    else:
                        diff_pct = abs(gt_base - pred_base) / abs(gt_base)
                        if diff_pct <= tolerance:
                            text_matches, _ = check_text_overlap(ground_truth, predicted)
                            if text_matches:
                                found_match = True
                                break

                if found_match:
                    matched_gt.append(gt_val)
                else:
                    unmatched_gt.append(gt_val)

            if len(matched_gt) == len(gt_numbers):
                return True, f"List match: All {len(gt_numbers)} numbers found in prediction"
            else:
                return False, f"List mismatch: Found {len(matched_gt)}/{len(gt_numbers)} numbers. Missing: {unmatched_gt}"

        else:
            gt_val, gt_context = gt_numbers[0]

            try:
                gt_base, gt_unit = normalize_number_with_units(gt_val, gt_context)
            except Exception as e:
                raise ValueError(f"Failed to normalize GT number: {e}") from e

            gt_has_text, _ = has_significant_text(ground_truth)
            should_filter_years = not (is_likely_year(gt_val) or gt_has_text)

            best_match = None
            best_diff = float('inf')
            best_pred_info = None
            unit_mismatches = []

            for pred_val, pred_context in pred_numbers:
                if should_filter_years and is_likely_year(pred_val):
                    continue

                try:
                    pred_base, pred_unit = normalize_number_with_units(pred_val, pred_context)
                except Exception as e:
                    raise ValueError(f"Failed to normalize prediction number: {e}") from e

                if not units_compatible(gt_unit, pred_unit):
                    unit_mismatches.append((pred_base, pred_unit))
                    continue

                if gt_base == 0:
                    if pred_base == 0:
                        text_matches, text_rationale = check_text_overlap(ground_truth, predicted)
                        if text_matches:
                            return True, f"Exact match: Found 0 in response. {text_rationale}"
                    continue

                diff_pct = abs(gt_base - pred_base) / abs(gt_base)

                if diff_pct < best_diff:
                    best_diff = diff_pct
                    best_match = pred_base
                    best_pred_info = (pred_base, pred_unit)

                if diff_pct <= tolerance:
                    text_matches, text_rationale = check_text_overlap(ground_truth, predicted)
                    if not text_matches:
                        continue

                    return True, f"Numerical match: GT={gt_base} ({gt_unit or 'no unit'}), Pred={pred_base} ({pred_unit or 'no unit'}), Diff={diff_pct*100:.2f}%. {text_rationale}"

            if best_match is not None and best_pred_info is not None:
                return False, f"No match: GT={gt_base} ({gt_unit or 'no unit'}), Closest={best_pred_info[0]} ({best_pred_info[1] or 'no unit'}), Diff={best_diff*100:.2f}%"
            if unit_mismatches:
                pred_units = [unit or "no unit" for _, unit in unit_mismatches[:5]]
                return False, f"No match: explicit unit mismatch. GT unit={gt_unit or 'no unit'}, Pred units={pred_units}"
            return False, f"No valid numbers found in prediction (filtered out years: {[n for n, _ in pred_numbers[:5]]})"

    gt_clean = ground_truth.strip().lower().strip('"').strip("'")
    pred_clean = predicted.strip().lower().strip('"').strip("'")

    gt_clean = re.sub(r'\s+', ' ', re.sub(r'\([^)]*\)', '', gt_clean)).strip()
    pred_clean = re.sub(r'\s+', ' ', re.sub(r'\([^)]*\)', '', pred_clean)).strip()

    if gt_clean in pred_clean:
        return True, f"Text match: '{ground_truth}' found in prediction"

    if gt_clean == pred_clean:
        return True, "Exact text match"

    return False, f"No match found. GT: '{ground_truth[:100]}', Pred: '{predicted[:100]}'"


def _normalize_direct_text_answer(text: str) -> str:
    cleaned = text.strip().lower().strip('"').strip("'")
    cleaned = re.sub(r'\([^)]*\)', '', cleaned).strip()
    cleaned = re.sub(r'\s+', ' ', cleaned)
    return cleaned


def _is_direct_answer_only(ground_truth: str, predicted: str) -> tuple[bool, str]:
    predicted = predicted.strip()
    if not predicted:
        return False, "Predicted answer is empty"

    nonempty_lines = [line for line in predicted.splitlines() if line.strip()]
    if len(nonempty_lines) > 1:
        return False, "Predicted answer spans multiple non-empty lines"

    if len(predicted) > 250:
        return False, "Predicted answer is too long to be a direct answer"

    gt_numbers_with_context = extract_numbers_with_context(ground_truth)
    pred_numbers_with_context = extract_numbers_with_context(predicted)

    gt_numbers = [(num, ctx) for num, ctx, _, _ in gt_numbers_with_context]
    pred_numbers = [(num, ctx) for num, ctx, _, _ in pred_numbers_with_context]

    if gt_numbers:
        if len(pred_numbers) != len(gt_numbers):
            return False, (
                "Predicted answer must contain exactly the expected answer numbers"
            )

        gt_has_text, _ = has_significant_text(ground_truth)
        pred_has_text, pred_text = has_significant_text(predicted)
        if not gt_has_text and pred_has_text:
            return (
                False,
                "Predicted answer has prose outside the answer value "
                f"(extra text after removing numbers/units: '{pred_text}')",
            )

        return True, "Direct answer only"

    if pred_numbers:
        return False, "Prediction contains numbers but ground truth is text-only"

    if _normalize_direct_text_answer(ground_truth) != _normalize_direct_text_answer(predicted):
        return False, "Text answer must match the expected answer text exactly"

    return True, "Direct answer only"


def score_answer(ground_truth: str, predicted: str, tolerance: float = 0.00) -> float:
    """Score the answer using robust fuzzy matching."""
    try:
        predicted, _ = extract_final_answer_from_xml(predicted)
    except Exception:
        return 0.0

    try:
        ok, _ = _is_direct_answer_only(ground_truth, predicted)
        if not ok:
            return 0.0

        is_correct, _ = fuzzy_match_answer(ground_truth, predicted, tolerance)
    except Exception:
        return 0.0

    return 1.0 if is_correct else 0.0
