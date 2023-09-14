import { ChatWindow } from "@/components/ChatWindow";

export default function AgentsPage() {
  const InfoCard = (
    <div className="p-4 md:p-8 rounded bg-[#25252d] w-full max-h-[85%] overflow-hidden">
      <h1 className="text-3xl md:text-4xl mb-4">
        Strukturiere deine Eingabe 🧱
      </h1>
    </div>
  );
  return (
    <ChatWindow
      endpoint="api/chat/structured_output"
      emptyStateComponent={InfoCard}
      placeholder={`Egal, was Sie hier eingeben, ich werde die Eingabe mit der gleichen Struktur zurückgeben!"`}
      emoji="🧱"
      titleText="Strukturierte Ausgabe"
    ></ChatWindow>
  );
}
