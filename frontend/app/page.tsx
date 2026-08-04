import Link from "next/link";

export default function Home() {
  return (
    <main className="p-6">
      <h1 className="text-2xl font-semibold">Support Ticket Intelligence</h1>
      <p className="mt-2">Start here:</p>
      <div className="mt-4 flex gap-3">
        <Link className="underline" href="/dashboard">Dashboard</Link>
        <Link className="underline" href="/inbox">Inbox</Link>
      </div>
    </main>
  );
}
