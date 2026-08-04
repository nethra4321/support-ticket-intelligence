"use client";

import { useEffect, useState } from "react";

type ModelName = "mistral" | "gpt2" | "gpt5" | "claude";

type TicketDetail = {
  ticket_id: string;
  created_at: string;
  customer_id: string;
  text: string;
  label: string;
  model: string;
  summary: string;
  suggested_reply: string;
};

const BACKEND_URL =
  process.env.NEXT_PUBLIC_BACKEND_URL || "http://localhost:8000";

const modelLabels: Record<ModelName, string> = {
  mistral: "Mistral",
  gpt2: "GPT-2",
  gpt5: "GPT-5",
  claude: "Claude Opus",
};

export default function AiControls({ ticketId }: { ticketId: string }) {
  const [viewModel, setViewModel] = useState<ModelName>("mistral");
  const [loading, setLoading] = useState<ModelName | null>(null);
  const [ticket, setTicket] = useState<TicketDetail | null>(null);
  const [error, setError] = useState<string | null>(null);

  async function load(model: ModelName) {
    setError(null);

    try {
      const res = await fetch(
        `${BACKEND_URL}/tickets/${encodeURIComponent(
          ticketId
        )}?model=${model}`,
        {
          cache: "no-store",
        }
      );

      const text = await res.text();

      let data: TicketDetail | { detail?: string } | null = null;

      try {
        data = text ? JSON.parse(text) : null;
      } catch {
        throw new Error("Backend returned an invalid response.");
      }

      if (!res.ok) {
        throw new Error(
          data && "detail" in data && data.detail
            ? data.detail
            : `Failed to load ticket (${res.status})`
        );
      }

      setTicket(data as TicketDetail);
    } catch (err) {
      const message =
        err instanceof Error ? err.message : "Failed to load AI output.";

      console.error("Load failed:", err);
      setError(message);
    }
  }

  async function generate(model: ModelName) {
    setLoading(model);
    setError(null);

    try {
      const res = await fetch(
        `${BACKEND_URL}/tickets/${encodeURIComponent(
          ticketId
        )}/generate?model=${model}`,
        {
          method: "POST",
        }
      );

      const text = await res.text();

      let data: { detail?: string } | null = null;

      try {
        data = text ? JSON.parse(text) : null;
      } catch {
        throw new Error("Backend returned an invalid response.");
      }

      if (!res.ok) {
        throw new Error(
          data?.detail || `Generation failed (${res.status})`
        );
      }

      setViewModel(model);
      await load(model);
    } catch (err) {
      const message =
        err instanceof Error ? err.message : "Failed to generate AI output.";

      console.error("Generate failed:", err);
      setError(message);
    } finally {
      setLoading(null);
    }
  }

  useEffect(() => {
    load(viewModel);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [viewModel, ticketId]);

  return (
    <div className="mt-4">
      <div className="flex flex-wrap items-center gap-2">
        {(Object.keys(modelLabels) as ModelName[]).map((model) => (
          <button
            key={model}
            type="button"
            onClick={() => generate(model)}
            disabled={loading !== null}
            className="rounded border px-3 py-2 text-sm hover:bg-gray-50 disabled:cursor-not-allowed disabled:opacity-50"
          >
            {loading === model ? "Generating..." : modelLabels[model]}
          </button>
        ))}

        <div className="ml-auto flex items-center gap-2 text-sm">
          <span className="text-gray-600">View:</span>

          <select
            value={viewModel}
            onChange={(event) =>
              setViewModel(event.target.value as ModelName)
            }
            className="rounded border px-2 py-1"
          >
            {(Object.keys(modelLabels) as ModelName[]).map((model) => (
              <option key={model} value={model}>
                {modelLabels[model]}
              </option>
            ))}
          </select>
        </div>
      </div>

      {error && (
        <div className="mt-4 rounded border border-red-300 bg-red-50 p-3 text-sm text-red-700">
          {error}
        </div>
      )}

      <div className="mt-4 rounded border p-3">
        <div className="text-xs text-gray-600">
          Showing output from: {modelLabels[viewModel]}
        </div>

        <div className="mt-3">
          <div className="font-medium">Summary</div>
          <div className="mt-1 text-sm">
            {ticket?.summary || "Not generated yet."}
          </div>
        </div>

        <div className="mt-4">
          <div className="font-medium">Suggested Reply</div>
          <div className="mt-1 whitespace-pre-wrap text-sm">
            {ticket?.suggested_reply || "Not generated yet."}
          </div>
        </div>
      </div>
    </div>
  );
}