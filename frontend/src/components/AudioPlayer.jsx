import React, { useEffect } from 'react';

export default function AudioPlayer({ audioQueue, setAudioQueue }) {
  useEffect(() => {
    if (audioQueue.length > 0) {
      const audioData = audioQueue[0];
      const audio = new Audio(`data:audio/wav;base64,${audioData}`);
      audio.play();

      audio.onended = () => {
        setAudioQueue((q) => q.slice(1));
      };
    }
  }, [audioQueue]);

  return null;
}
