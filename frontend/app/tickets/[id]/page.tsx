import AiControls from "./AiControls";

type TicketDetail = {
  ticket_id: string;
  created_at: string;
  customer_id: string;
  text: string;
  label: string;
  summary: string;
  suggested_reply: string;
};

async function fetchTicket(id: string): Promise<TicketDetail> {
  const base =
    process.env.NEXT_PUBLIC_BACKEND_URL || "http://localhost:8000";

  const res = await fetch(
    `${base}/tickets/${encodeURIComponent(id)}`,
    {
      cache: "no-store",
    }
  );

  if (!res.ok) {
    const message = await res.text().catch(() => "");
    throw new Error(
      `Failed to fetch ticket (${res.status}): ${message}`
    );
  }

  return res.json();
}

export default async function TicketPage({
  params,
}: {
  params: Promise<{ id: string }>;
}) {
  const { id } = await params;
  const ticket = await fetchTicket(id);

  return (
    <main className="p-6">
      <h1 className="text-2xl font-semibold">
        Ticket {ticket.ticket_id}
      </h1>

      <div className="mt-2 text-sm text-gray-600">
        {ticket.created_at}
      </div>

      <div className="mt-6 grid grid-cols-1 gap-4 md:grid-cols-2">
        <section className="rounded border p-4">
          <h2 className="font-semibold">Ticket Text</h2>

          <p className="mt-2 whitespace-pre-wrap">
            {ticket.text}
          </p>
        </section>

        <section className="rounded border p-4">
          <h2 className="font-semibold">AI Analysis</h2>

          <div className="mt-2 text-sm">
            Category: {ticket.label}
          </div>

          <AiControls ticketId={ticket.ticket_id} />
        </section>
      </div>

      <div className="mt-6">
        <a className="underline" href="/inbox">
          ← Back to Inbox
        </a>
      </div>
    </main>
  );
}