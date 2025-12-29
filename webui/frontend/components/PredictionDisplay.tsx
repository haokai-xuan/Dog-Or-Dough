import React, { useState, useEffect } from 'react'
import CountUp from 'react-countup';

type PredictionProps = {
    dog: number;
    dough: number;
}

const PredictionDisplay: React.FC<PredictionProps> = ({dog, dough}) => {
  const [animatedDog, setAnimatedDog] = useState(0);
  const [animatedDough, setAnimatedDough] = useState(0);

  useEffect(() => {
    const timer = setTimeout(() => {
      setAnimatedDog(dog);
      setAnimatedDough(dough);
    }, 100);

    return () => clearTimeout(timer);
  }, [dog, dough]);

  return (
    <div className="mt-4 w-64 mx-auto space-y-2">
      <div className="flex justify-between">
        <span>Dog 🐕</span>
        <span>
          <CountUp end={animatedDog} duration={1} decimals={2} />%
        </span>
      </div>
      <div className="w-full bg-gray-200 rounded-full h-4">
        <div 
          className="bg-purple-500 h-4 rounded-full transition-all duration-1000 ease-out" 
          style={{width: `${animatedDog}%`}}
        ></div>
      </div>

      <div className="flex justify-between">
        <span>Dough 🍞</span>
        <span>
          <CountUp end={animatedDough} duration={1} decimals={2} />%
        </span>
      </div>
      <div className="w-full bg-gray-200 rounded-full h-4">
        <div 
          className="bg-yellow-400 h-4 rounded-full transition-all duration-1000 ease-out" 
          style={{width: `${animatedDough}%`}}
        ></div>
      </div>
    </div>
  )
}

export default PredictionDisplay