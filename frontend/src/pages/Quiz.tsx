import { useState, useEffect } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { Maximize, Minimize } from 'lucide-react';

export default function Quiz() {
    const [showAnswer, setShowAnswer] = useState(false);
    const [isFullscreen, setIsFullscreen] = useState(false);

    const handleAnswer = () => {
        setShowAnswer(true);
    };

    const toggleFullscreen = () => {
        if (!document.fullscreenElement) {
            document.documentElement.requestFullscreen();
            setIsFullscreen(true);
        } else {
            if (document.exitFullscreen) {
                document.exitFullscreen();
                setIsFullscreen(false);
            }
        }
    };

    // Handle fullscreen change events
    useEffect(() => {
        const handleFullscreenChange = () => {
            setIsFullscreen(!!document.fullscreenElement);
        };

        document.addEventListener('fullscreenchange', handleFullscreenChange);

        // Keyboard shortcut: F key to toggle fullscreen
        const handleKeyPress = (e: KeyboardEvent) => {
            if (e.key === 'f' || e.key === 'F') {
                toggleFullscreen();
            }
        };

        document.addEventListener('keydown', handleKeyPress);

        return () => {
            document.removeEventListener('fullscreenchange', handleFullscreenChange);
            document.removeEventListener('keydown', handleKeyPress);
        };
    }, []);

    return (
        <div className="min-h-screen bg-white flex items-center justify-center p-8 relative">
            <style>{`
        @keyframes gradient-shift {
          0% {
            background-position: 0% 50%;
          }
          50% {
            background-position: 100% 50%;
          }
          100% {
            background-position: 0% 50%;
          }
        }
        .infinia-gradient {
          background: linear-gradient(
            90deg,
            #3b82f6,
            #8b5cf6,
            #ec4899,
            #f59e0b,
            #10b981,
            #3b82f6
          );
          background-size: 200% 200%;
          animation: gradient-shift 3s ease infinite;
          -webkit-background-clip: text;
          background-clip: text;
          -webkit-text-fill-color: transparent;
        }
      `}</style>

            {/* Fullscreen Toggle Button */}
            <motion.button
                initial={{ opacity: 0 }}
                animate={{ opacity: 1 }}
                transition={{ delay: 0.5 }}
                whileHover={{ scale: 1.1 }}
                whileTap={{ scale: 0.9 }}
                onClick={toggleFullscreen}
                className="fixed top-6 right-6 z-50 p-4 rounded-full bg-gray-800 text-white hover:bg-gray-700 transition-colors shadow-lg"
                title={isFullscreen ? "Exit Fullscreen (F)" : "Enter Fullscreen (F)"}
            >
                {isFullscreen ? (
                    <Minimize className="w-6 h-6" />
                ) : (
                    <Maximize className="w-6 h-6" />
                )}
            </motion.button>

            <AnimatePresence mode="wait">
                {!showAnswer ? (
                    <motion.div
                        key="question"
                        initial={{ opacity: 0, y: 20 }}
                        animate={{ opacity: 1, y: 0 }}
                        exit={{ opacity: 0, y: -20 }}
                        className="max-w-5xl w-full text-center"
                    >
                        {/* Subtitle */}
                        <motion.p
                            initial={{ opacity: 0 }}
                            animate={{ opacity: 1 }}
                            transition={{ delay: 0.2 }}
                            className="text-2xl md:text-3xl text-gray-600 mb-8 italic"
                        >
                            Think Alex is not in the Room
                        </motion.p>

                        {/* Main Question */}
                        <motion.h1
                            initial={{ opacity: 0, scale: 0.9 }}
                            animate={{ opacity: 1, scale: 1 }}
                            transition={{ delay: 0.4 }}
                            className="text-6xl md:text-7xl lg:text-8xl font-bold text-gray-900 mb-16 leading-tight"
                        >
                            Can Infinia Improve my Model Performance?
                        </motion.h1>

                        {/* Answer Buttons */}
                        <motion.div
                            initial={{ opacity: 0 }}
                            animate={{ opacity: 1 }}
                            transition={{ delay: 0.6 }}
                            className="flex flex-col md:flex-row gap-6 justify-center items-center"
                        >
                            <motion.button
                                whileHover={{ scale: 1.05 }}
                                whileTap={{ scale: 0.95 }}
                                onClick={handleAnswer}
                                className="px-12 py-6 text-4xl md:text-5xl font-bold rounded-2xl bg-green-500 text-white hover:bg-green-600 transition-colors shadow-lg min-w-[200px]"
                            >
                                Yes
                            </motion.button>

                            <motion.button
                                whileHover={{ scale: 1.05 }}
                                whileTap={{ scale: 0.95 }}
                                onClick={handleAnswer}
                                className="px-12 py-6 text-4xl md:text-5xl font-bold rounded-2xl bg-red-500 text-white hover:bg-red-600 transition-colors shadow-lg min-w-[200px]"
                            >
                                No
                            </motion.button>

                            <motion.button
                                whileHover={{ scale: 1.05 }}
                                whileTap={{ scale: 0.95 }}
                                onClick={handleAnswer}
                                className="px-12 py-6 text-4xl md:text-5xl font-bold rounded-2xl bg-gray-500 text-white hover:bg-gray-600 transition-colors shadow-lg min-w-[200px]"
                            >
                                I don't know
                            </motion.button>

                            <motion.button
                                whileHover={{ scale: 1.05 }}
                                whileTap={{ scale: 0.95 }}
                                onClick={handleAnswer}
                                className="px-12 py-6 text-4xl md:text-5xl font-bold rounded-2xl bg-gradient-to-r from-purple-500 to-pink-500 text-white hover:from-purple-600 hover:to-pink-600 transition-colors shadow-lg min-w-[200px]"
                            >
                                CALL SEVEN
                            </motion.button>
                        </motion.div>
                    </motion.div>
                ) : (
                    <motion.div
                        key="answer"
                        initial={{ opacity: 0, scale: 0.8 }}
                        animate={{ opacity: 1, scale: 1 }}
                        exit={{ opacity: 0, scale: 0.8 }}
                        className="max-w-6xl w-full text-center"
                    >
                        {/* Big Answer */}
                        <motion.div
                            initial={{ opacity: 0, y: 30 }}
                            animate={{ opacity: 1, y: 0 }}
                            transition={{ delay: 0.2 }}
                            className="mb-12"
                        >
                            <h1 className="text-8xl md:text-9xl font-bold bg-gradient-to-r from-blue-600 via-purple-600 to-pink-600 bg-clip-text text-transparent mb-6">
                                101%
                            </h1>
                            <p className="text-4xl md:text-5xl lg:text-6xl font-bold leading-tight">
                                <span className="text-gray-900">Only </span>
                                <span className="infinia-gradient">INFINIA</span>
                                <span className="text-gray-900"> Data Intelligence Platform</span>
                                <br />
                                <span className="text-gray-900">can Improve Model Accuracy</span>
                            </p>
                        </motion.div>

                        {/* Quote */}
                        <motion.div
                            initial={{ opacity: 0, y: 30 }}
                            animate={{ opacity: 1, y: 0 }}
                            transition={{ delay: 0.5 }}
                            className="bg-gradient-to-br from-blue-50 to-purple-50 rounded-3xl p-12 border-4 border-blue-200 shadow-2xl"
                        >
                            <div className="text-3xl md:text-4xl lg:text-5xl text-gray-800 leading-relaxed space-y-6">
                                <p className="font-semibold italic">
                                    "If AI is the brain, storage is the memory.
                                </p>
                                <p className="font-semibold italic">
                                    Better memory leads to better judgment.
                                </p>
                                <p className="font-bold italic text-blue-600">
                                    Storage influences accuracy, not just speed."
                                </p>
                            </div>
                        </motion.div>

                        {/* Try Again Button */}
                        <motion.button
                            initial={{ opacity: 0 }}
                            animate={{ opacity: 1 }}
                            transition={{ delay: 0.8 }}
                            whileHover={{ scale: 1.05 }}
                            whileTap={{ scale: 0.95 }}
                            onClick={() => setShowAnswer(false)}
                            className="mt-12 px-10 py-5 text-2xl md:text-3xl font-bold rounded-xl bg-blue-600 text-white hover:bg-blue-700 transition-colors shadow-lg"
                        >
                            Ask Again
                        </motion.button>
                    </motion.div>
                )}
            </AnimatePresence>
        </div>
    );
}
