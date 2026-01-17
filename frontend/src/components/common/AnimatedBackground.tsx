/**
 * Animated background with floating gradient lights
 * Inspired by the Ozera logo aesthetic
 */

import React from 'react'

export const AnimatedBackground: React.FC = () => {
  return (
    <div className="animated-background">
      <div className="gradient-orb orb-1"></div>
      <div className="gradient-orb orb-2"></div>
      <div className="gradient-orb orb-3"></div>
      <div className="gradient-orb orb-4"></div>
      <div className="gradient-orb orb-5"></div>

      <style>{`
        .animated-background {
          position: fixed;
          top: 0;
          left: 0;
          width: 100%;
          height: 100%;
          background: #000000;
          overflow: hidden;
          z-index: 1;
          pointer-events: none;
        }

        .gradient-orb {
          position: absolute;
          filter: blur(120px);
          opacity: 0.4;
          animation-timing-function: ease-in-out;
          animation-iteration-count: infinite;
          animation-direction: alternate;
          will-change: transform;
          mix-blend-mode: screen;
        }

        .orb-1 {
          width: 1000px;
          height: 1000px;
          background: radial-gradient(circle, rgba(79, 70, 229, 1) 0%, rgba(79, 70, 229, 0.6) 30%, transparent 45%);
          top: -10%;
          right: -10%;
          animation: float1 27s infinite;
        }

        .orb-2 {
          width: 900px;
          height: 900px;
          background: radial-gradient(circle, rgb(51, 234, 140) 0%, rgba(51, 234, 124, 0.6) 30%, transparent 45%);
          bottom: -10%;
          left: -10%;
          animation: float2 30s infinite;
        }

        .orb-3 {
          width: 1100px;
          height: 1100px;
          background: radial-gradient(circle, rgba(59, 130, 246, 1) 0%, rgba(59, 130, 246, 0.5) 30%, transparent 45%);
          top: 50%;
          left: 50%;
          animation: float3 33s infinite;
        }

        .orb-4 {
          width: 950px;
          height: 950px;
          background: radial-gradient(circle, rgba(99, 102, 241, 1) 0%, rgba(99, 102, 241, 0.6) 30%, transparent 45%);
          top: -10%;
          left: -10%;
          animation: float4 28.5s infinite;
        }

        .orb-5 {
          width: 1050px;
          height: 1050px;
          background: radial-gradient(circle, rgba(124, 58, 237, 1) 0%, rgba(124, 58, 237, 0.5) 30%, transparent 45%);
          bottom: -10%;
          right: -10%;
          animation: float5 31.5s infinite;
        }

        @keyframes float1 {
          0% {
            transform: translate(0, 0) scale(1);
          }
          25% {
            transform: translate(-60vw, 40vh) scale(1.1);
          }
          50% {
            transform: translate(-80vw, 80vh) scale(1.3);
          }
          75% {
            transform: translate(-40vw, 50vh) scale(0.9);
          }
          100% {
            transform: translate(0, 0) scale(1);
          }
        }

        @keyframes float2 {
          0% {
            transform: translate(0, 0) scale(1);
          }
          25% {
            transform: translate(50vw, -30vh) scale(1.2);
          }
          50% {
            transform: translate(90vw, -60vh) scale(0.8);
          }
          75% {
            transform: translate(45vw, -20vh) scale(1.15);
          }
          100% {
            transform: translate(0, 0) scale(1);
          }
        }

        @keyframes float3 {
          0% {
            transform: translate(-50%, -50%) scale(1);
          }
          25% {
            transform: translate(calc(-50% - 40vw), calc(-50% - 35vh)) scale(0.85);
          }
          50% {
            transform: translate(calc(-50% + 45vw), calc(-50% + 40vh)) scale(1.25);
          }
          75% {
            transform: translate(calc(-50% - 20vw), calc(-50% + 20vh)) scale(1.05);
          }
          100% {
            transform: translate(-50%, -50%) scale(1);
          }
        }

        @keyframes float4 {
          0% {
            transform: translate(0, 0) scale(1);
          }
          25% {
            transform: translate(70vw, 30vh) scale(1.1);
          }
          50% {
            transform: translate(95vw, 70vh) scale(0.9);
          }
          75% {
            transform: translate(60vw, 40vh) scale(1.2);
          }
          100% {
            transform: translate(0, 0) scale(1);
          }
        }

        @keyframes float5 {
          0% {
            transform: translate(0, 0) scale(1);
          }
          25% {
            transform: translate(-50vw, -40vh) scale(1.15);
          }
          50% {
            transform: translate(-85vw, -75vh) scale(0.95);
          }
          75% {
            transform: translate(-40vw, -35vh) scale(1.08);
          }
          100% {
            transform: translate(0, 0) scale(1);
          }
        }
      `}</style>
    </div>
  )
}

export default AnimatedBackground
