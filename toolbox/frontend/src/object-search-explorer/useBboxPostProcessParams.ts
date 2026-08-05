import { useEffect, useState } from "react";

import {
  DEFAULT_BBOX_POST_PROCESS,
  readStoredBboxPostProcess,
  storeBboxPostProcess,
  type BboxPostProcessParams,
} from "./bboxPostProcess";

export function useBboxPostProcessParams() {
  const [params, setParams] = useState<BboxPostProcessParams>(readStoredBboxPostProcess);

  useEffect(() => {
    storeBboxPostProcess(params);
  }, [params]);

  const resetParams = () => setParams(DEFAULT_BBOX_POST_PROCESS);

  return { params, setParams, resetParams };
}
