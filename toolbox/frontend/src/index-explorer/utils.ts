import type { IndexObjectRecord } from "./types";

export function objectMatchesOcrFilters(
  item: IndexObjectRecord,
  filters: {
    query: string;
    sourceFilter: number | null;
    requireKey: boolean;
    candidateOnly: boolean;
    assignedOnly: boolean;
  },
): boolean {
  if (filters.query) {
    const haystack = `${item.ocr_text} ${item.ocr_tokens} ${item.ocr_key}`.toLowerCase();
    if (!haystack.includes(filters.query.toLowerCase())) {
      return false;
    }
  }
  if (filters.sourceFilter !== null) {
    const sourcePrefix = `${filters.sourceFilter} -`;
    if (!item.ocr_source.startsWith(sourcePrefix)) {
      return false;
    }
  }
  if (filters.requireKey && !item.ocr_key.trim()) {
    return false;
  }
  if (filters.candidateOnly && !item.ocr_candidate) {
    return false;
  }
  if (filters.assignedOnly && !item.ocr_assigned) {
    return false;
  }
  return true;
}

export function parseOcrSourceFilter(label: string): number | null {
  if (label === "all") {
    return null;
  }
  const match = /^(\d+)\s*-/.exec(label);
  return match ? Number(match[1]) : null;
}
