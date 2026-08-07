export type ReviewStatus = "true_positive" | "false_positive";

export type DetectionReview = {
  targetId: number;
  status: ReviewStatus;
};

type DetectionReviewResponse = {
  detection_reviews?: unknown;
};

function reviewUrl(mapId: string, suffix = ""): string {
  return `/ui/api/maps/${encodeURIComponent(mapId)}/review-annotations${suffix}`;
}

async function errorDetail(response: Response): Promise<string> {
  const text = await response.text();
  if (!text) {
    return `HTTP ${response.status}`;
  }
  try {
    const parsed = JSON.parse(text) as { detail?: unknown };
    return typeof parsed.detail === "string" ? parsed.detail : text;
  } catch {
    return text;
  }
}

function isReviewStatus(value: unknown): value is ReviewStatus {
  return value === "true_positive" || value === "false_positive";
}

export async function fetchDetectionReviews(
  mapId: string,
  query: string,
): Promise<DetectionReview[]> {
  const params = new URLSearchParams({ query });
  const response = await fetch(`${reviewUrl(mapId)}?${params.toString()}`, {
    headers: { Accept: "application/json" },
  });
  if (!response.ok) {
    throw new Error(await errorDetail(response));
  }
  const data = (await response.json()) as DetectionReviewResponse;
  if (!Array.isArray(data.detection_reviews)) {
    return [];
  }
  return data.detection_reviews.flatMap((raw) => {
    if (!raw || typeof raw !== "object") {
      return [];
    }
    const item = raw as Record<string, unknown>;
    const targetId = Number(item.target_id);
    if (
      item.target_type !== "object" ||
      !Number.isInteger(targetId) ||
      !isReviewStatus(item.status)
    ) {
      return [];
    }
    return [{ targetId, status: item.status }];
  });
}

export async function setDetectionReview(
  mapId: string,
  query: string,
  targetId: number,
  status: ReviewStatus | null,
): Promise<void> {
  const response = await fetch(reviewUrl(mapId, "/detection-review"), {
    method: status === null ? "DELETE" : "POST",
    headers: {
      Accept: "application/json",
      "Content-Type": "application/json",
    },
    body: JSON.stringify({
      target_type: "object",
      target_id: targetId,
      query,
      ...(status === null ? {} : { status }),
    }),
  });
  if (!response.ok) {
    throw new Error(await errorDetail(response));
  }
}
