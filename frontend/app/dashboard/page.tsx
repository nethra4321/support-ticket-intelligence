export default function DashboardPage() {
  return (
    <main className="p-6">
      <h1 className="text-2xl font-semibold">Dashboard</h1>
      <div className="mt-4 grid grid-cols-1 gap-4 md:grid-cols-3">
        <div className="rounded border p-4">
          <div className="text-sm text-gray-600">Tickets loaded</div>
          <div className="mt-2 text-2xl font-semibold">From Snowflake</div>
        </div>
        <div className="rounded border p-4">
          <div className="text-sm text-gray-600">Model</div>
          <div className="mt-2 text-2xl font-semibold">Not yet</div>
        </div>
        <div className="rounded border p-4">
          <div className="text-sm text-gray-600">GPT</div>
          <div className="mt-2 text-2xl font-semibold">Next step</div>
        </div>
      </div>
    </main>
  );
}
