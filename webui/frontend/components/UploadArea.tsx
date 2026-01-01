"use client";

import Image from "next/image"
import { useCallback, useEffect, useState } from "react"
import { FileRejection, useDropzone } from "react-dropzone"
import PredictionDisplay from "./PredictionDisplay";
import { Link, Element } from "react-scroll";

type PreviewFile = File & {preview: string}

const UploadArea = () => {
  const [file, setFile] = useState<PreviewFile | null>(null)
  const [fileTooLarge, setFileTooLarge] = useState<boolean>(false)

  const onDrop = useCallback((acceptedFiles: File[]) => {
    if (!acceptedFiles.length) return

    const f = Object.assign(acceptedFiles[0], {
      preview: URL.createObjectURL(acceptedFiles[0])
    })
    setFileTooLarge(false)
    setFile(f)
  }, [])

  const onDropRejected = useCallback((acceptedFiles: FileRejection[]) => {
    setFile(null)
    setFileTooLarge(true)
  }, [])

  useEffect(() => {
    setPrediction(null)
    return () => {
      if (file) URL.revokeObjectURL(file.preview)
    }
  }, [file])

  const removeFile = () => {
    setFile(null)
  }

  const {getRootProps, getInputProps, isDragActive} = useDropzone({
    onDrop,
    onDropRejected,
    accept: {
      'image/jpeg': [],
      'image/png': [],
      'image/webp': [],
    },
    maxFiles: 1,
    maxSize: 4500000 // 4.5MB
  })

  useEffect(() => {
    const handlePaste = (e: ClipboardEvent) => {
      const items = e.clipboardData?.items
      if (!items) return
      for (let i = 0; i < items.length; i++) {
        if (items[i].type.indexOf("image") !== -1) {
          const blob = items[i].getAsFile()
          if (!blob) return

          if (blob.size > 4500000) {
            setFile(null)
            setFileTooLarge(true)
            return
          }

          const file = new File([blob], "pasted-img.png", {type: blob.type})
          const f = Object.assign(file, {
            preview: URL.createObjectURL(file)
          })
          setFileTooLarge(false)
          setFile(f)
          e.preventDefault()
          return
        }
      }
    }

    window.addEventListener("paste", handlePaste)
    return () => window.removeEventListener("paste", handlePaste)
  }, [])

  const [prediction, setPrediction] = useState<{dog: number; dough: number} | null>(null)
  const [handlingInference, setHandlingInference] = useState<boolean>(false)
  const [error, setError] = useState<string | null>(null)
  const handleInference = async () => {
    if (!file) return;

    setHandlingInference(true)
    setError(null)
    const formData = new FormData()
    formData.append("file", file)
    
    try {
      const res = await fetch("/api/predict", {
        method: "POST",
        body: formData,
      })

      const data = await res.json();

      if (res.status === 429) {
        setError(data.error + (data.retryAfter ? `(${data.retryAfter}s)` : ""))
        return
      }
      if (data.error) {
        setError(data.error)
        return
      }

      const dog = Number((data["dog"] * 100).toFixed(2))
      const dough = Number((data["dough"] * 100).toFixed(2))
      setPrediction({ dog: dog, dough: dough });
      // console.log("Server response:", data);
    }
    catch (err) {
      setError("Request failed, please try again.")
    }
    finally {
      setHandlingInference(false)
    }
  }
  
  return (
    <>
      <div
        {...getRootProps()}
        className="w-full mt-10 border-2 border-dashed border-gray-400 rounded-2xl p-10 text-center cursor-pointer transition hover:border-gray-600 h-64 flex flex-col justify-center items-center mx-auto max-w-md"
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
            <p className="text-purple-200 text-sm">or paste</p>
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
                  cursor-pointer
                ">
                x
              </button>
            </div>
          </div>

          <Link to="prediction-display" smooth={true} duration={500}>
          {
            handlingInference &&
            <button disabled className="bg-purple-400 hover:bg-purple-600 text-white font-semibold px-6 py-3 rounded-xl flex items-center">
              <svg
                className="mr-3 size-5 animate-spin"
                viewBox="0 0 24 24"
              >
                <circle
                  className="opacity-25"
                  cx="12"
                  cy="12"
                  r="10"
                  stroke="currentColor"
                  strokeWidth="4"
                ></circle>
                <path
                  className="opacity-75"
                  fill="currentColor"
                  d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"
                ></path>
              </svg>
              Inferencing...
            </button>
          }
          {
            !handlingInference &&
            <button onClick={handleInference} className="bg-purple-400 cursor-pointer hover:bg-purple-600 text-white font-semibold px-6 py-3 rounded-xl">
              Inference
            </button>
          }
          </Link>
        </div>
      }
      <Element name="prediction-display">
        {
          file && prediction &&
          <>
            <hr className="w-lg h-px mx-auto bg-gray-400 border-0 rounded-md my-4"/>
            <PredictionDisplay dog={prediction.dog} dough={prediction.dough} />
          </>
        }
      </Element>
      {
        fileTooLarge &&
        <div className="text-red-500 text-center">
          File limit exceeded (4.5MB)
        </div>
      }
      {
        error &&
        <div className="text-red-500 text-center">
          {error}
        </div>
      }
    </>
  )
}

export default UploadArea