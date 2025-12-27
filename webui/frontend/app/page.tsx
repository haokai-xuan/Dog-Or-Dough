import UploadArea from "@/components/UploadArea";
import Image from "next/image";

export const metadata = {
  title: "Dog or Dough"
}

export default function Home() {
  return (
    <main className="overflow-hidden min-h-screen px-6">
      <header className="pt-12">
        <h1 className="text-5xl font-extrabold text-center">
          <span className="text-purple-500">Dog</span> or <span className="text-yellow-400">Dough</span>
        </h1>
        <p className="text-center text-gray-500 mt-2">
          Upload an image and I’ll tell you whether it’s a dog… or dough 🐕🍞
        </p>
      </header>
      <UploadArea />
    </main>
  );
}
