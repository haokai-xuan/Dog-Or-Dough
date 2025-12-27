import React from 'react'

type PredictionProps = {
    dog: number;
    dough: number;
}

const PredictionDisplay: React.FC<PredictionProps> = ({dog, dough}) => {
  return (
    <div className="mt-4 w-64 mx-auto space-y-2">
        <div className="flex justify-between">
            <span>Dog 🐕</span>
            <span>{dog}%</span>
        </div>
        <div className="w-full bg-gray-200 rounded-full h-4">
            <div className="bg-purple-500 h-4 rounded-full" style={{width: `${dog}%`}}></div>
        </div>

        <div className="flex justify-between">
            <span>Dough 🍞</span>
            <span>{dough}%</span>
        </div>
        <div className="w-full bg-gray-200 rounded-full h-4">
            <div className="bg-yellow-400 h-4 rounded-full" style={{width: `${dough}%`}}></div>
        </div>
    </div>
  )
}

export default PredictionDisplay