"use client";

import Image from "next/image"
import { useCallback, useEffect, useState } from "react"
import { useDropzone } from "react-dropzone"
import PredictionDisplay from "./PredictionDisplay";

type PreviewFile = File & {preview: string}

const UploadArea = () => {
  const [file, setFile] = useState<PreviewFile | null>(null)
  const onDrop = useCallback((acceptedFiles: File[]) => {
    if (!acceptedFiles.length) return

    const f = Object.assign(acceptedFiles[0], {
      preview: URL.createObjectURL(acceptedFiles[0])
    })
    setFile(f)
  }, [])

  useEffect(() => {
    return () => {
      if (file) URL.revokeObjectURL(file.preview)
    }
  }, [file])

  const removeFile = () => {
    setFile(null)
  }

  const {getRootProps, getInputProps, isDragActive} = useDropzone({
    onDrop,
    accept: {
      'image/jpeg': [],
      'image/png': [],
      'image/webp': [],
    },
    maxFiles: 1,
    maxSize: 5000000 // 5MB
  })

  const [prediction, setPrediction] = useState<{dog: number; dough: number} | null>(null)
  const handleInference = () => {
    const mockDog = Math.floor(Math.random() * 100);
    const mockDough = 100 - mockDog;

    setPrediction({ dog: mockDog, dough: mockDough });
  }
  
  return (
    <>
      <div
        {...getRootProps()}
        className="mt-10 border-2 border-dashed border-gray-400 rounded-2xl p-10 text-center cursor-pointer transition hover:border-gray-600 h-64 flex flex-col justify-center items-center mx-auto max-w-md"
      >
        <input {...getInputProps()} />
        {isDragActive ? (
          <p className="text-purple-600 font-semibold text-lg">Release to upload</p>
        ) : (
          <>
            <span className="material-icons rounded-full p-2 border-2 border-purple-600 mb-2 text-purple-400">upload</span>
            <p>Drag image in this area</p>
            <p className="text-purple-200">or</p>
            <button className="bg-purple-400 p-3 rounded-2xl hover:bg-purple-600 cursor-pointer">
              Select Image
            </button>
            <p className="mt-5 text-purple-200 text-xs">Allowed file types: All image formats</p>
          </>
        )}
      </div>
      {file && 
        <div className="mt-6 flex flex-col items-center justify-center space-y-4">
          <div className="w-64 h-64 border border-gray-400 rounded-xl flex items-center justify-center">
            <div className="relative w-full h-64">
              <img src={file.preview} alt="preview" className="w-full h-full object-cover rounded-xl"/>          
              <button
                onClick={removeFile}
                className="
                  absolute -top-2 -right-2
                  bg-red-500 hover:bg-red-700
                  text-white
                  rounded-full
                  w-8 h-8
                  flex items-center justify-center
                  shadow-md
                ">
                x
              </button>
            </div>
          </div>

          <button onClick={handleInference} className="bg-purple-400 cursor-pointer hover:bg-purple-600 text-white font-semibold px-6 py-3 rounded-xl">
            Inference
          </button>
        </div>
      }
      {file && prediction && <PredictionDisplay dog={prediction.dog} dough={prediction.dough} />}
    </>
  )
}

export default UploadArea