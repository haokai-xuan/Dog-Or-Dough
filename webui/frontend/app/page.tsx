import UploadArea from "@/components/UploadArea";
import Image from "next/image";

export const metadata = {
  title: "Dog or Dough"
}

export default function Home() {
  return (
    <main className="overflow-hidden min-h-screen px-6 flex flex-col">
      <div className="fixed inset-0 -z-20 bg-linear-to-br from-gray-950 via-slate-900 to-indigo-950" />
      <header className="pt-12">
        <h1 className="text-5xl font-extrabold text-center">
          <span className="text-purple-500">Dog</span> or <span className="text-yellow-400">Dough</span>
        </h1>
        <p className="text-center text-gray-500 mt-2">
          Upload an image and I’ll tell you whether it’s a dog… or dough 🐕🍞
        </p>
      </header>
      <UploadArea />
      <div className="mt-auto">
        <footer className="mt-60 -mx-6 py-6 text-center text-sm text-gray-400 bg-gray-900">
          <a
            href="https://github.com/haokai-xuan/Dog-Or-Dough"
            target="_blank"
            className="text-gray-400 hover:text-gray-300"
          >
            Github
          </a>
        </footer>
      </div>
    </main>
  );
}
