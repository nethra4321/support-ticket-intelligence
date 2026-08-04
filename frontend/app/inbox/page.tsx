type Ticket = {
  ticket_id: string;
  created_at: string;
  customer_id: string;
  text: string;
  label: string;
  confidence: number;
};

async function fetchTickets(): Promise<Ticket[]> {
  const base = process.env.NEXT_PUBLIC_BACKEND_URL || "http://localhost:8000";
  const res = await fetch(`${base}/tickets?limit=50`, { cache: "no-store" });

  if (!res.ok) {
    throw new Error("Failed to fetch tickets");
  }

  return res.json();
}

export default async function InboxPage() {
  const tickets = await fetchTickets();

  return (
    <main className="p-6">
      <h1 className="text-2xl font-semibold">Inbox</h1>

      {tickets.length === 0 && (
        <p className="mt-4 text-gray-500">No tickets found.</p>
      )}

      <div className="mt-4 space-y-3">
        {tickets.map((t) => (
          <a
            key={t.ticket_id}
            href={`/tickets/${t.ticket_id}`}
            className="block rounded border p-4 hover:bg-gray-50"
          >
            <div className="flex justify-between text-sm text-gray-600">
              <span>{t.created_at}</span>
              <span>Customer: {t.customer_id}</span>
            </div>

            <div className="mt-1 font-medium line-clamp-1">
              {t.text}
            </div>

            <div className="mt-2 text-xs text-gray-600">
              Label: {t.label} • Confidence: {(t.confidence * 100).toFixed(0)}%
            </div>
          </a>
        ))}
      </div>
    </main>
  );
}
