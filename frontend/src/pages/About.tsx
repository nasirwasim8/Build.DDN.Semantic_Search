import { useEffect } from 'react'
import { motion } from 'framer-motion'
import { Image, Video, FileText, Zap, Database, ArrowRight, DollarSign, TrendingUp, Trophy, Search, FolderOpen, Brain } from 'lucide-react'

interface AboutPageProps {
  onStartDemo?: () => void
}

export default function AboutPage({ onStartDemo }: AboutPageProps) {
  useEffect(() => {
    const style = document.createElement('style')
    style.textContent = `
      @keyframes shimmer {
        0% { background-position: -200% 0; }
        100% { background-position: 200% 0; }
      }
      @keyframes pulse-ring {
        0% { transform: scale(0.8); opacity: 0.6; }
        50% { transform: scale(1); opacity: 0.3; }
        100% { transform: scale(0.8); opacity: 0.6; }
      }
      @media (prefers-reduced-motion: reduce) {
        @keyframes shimmer { 0%, 100% { background-position: 0 0; } }
        @keyframes pulse-ring { 0%, 100% { transform: scale(1); opacity: 0.4; } }
      }
    `
    document.head.appendChild(style)
    return () => { document.head.removeChild(style) }
  }, [])

  return (
    <div className="about-page">
      {/* Business Outcome Hero Section */}
      <section
        className="relative overflow-hidden"
        style={{
          background: 'linear-gradient(135deg, #0a0a0a 0%, #1a1a2e 50%, #16213e 100%)'
        }}
      >
        {/* Animated Background Elements */}
        <div className="absolute inset-0 overflow-hidden pointer-events-none">
          {/* Subtle Grid Pattern */}
          <div
            className="absolute inset-0 opacity-[0.03]"
            style={{
              backgroundImage: `linear-gradient(rgba(255,255,255,0.5) 1px, transparent 1px),
                               linear-gradient(90deg, rgba(255,255,255,0.5) 1px, transparent 1px)`,
              backgroundSize: '50px 50px'
            }}
          />
          {/* Gradient Spotlight Effect */}
          <div
            className="absolute"
            style={{
              width: '800px',
              height: '800px',
              background: 'radial-gradient(circle, rgba(252, 211, 77, 0.08) 0%, transparent 70%)',
              left: '50%',
              top: '50%',
              transform: 'translate(-50%, -50%)',
              filter: 'blur(40px)'
            }}
          />
        </div>

        <div className="relative z-10 max-w-[1400px] mx-auto px-6 py-24 md:py-32">
          {/* Executive Headline */}
          <motion.div
            initial={{ opacity: 0, y: 30 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.6, ease: [0.16, 1, 0.3, 1] }}
            className="text-center mb-12"
          >
            <div className="inline-block mb-6">
              <span
                className="px-6 py-2.5 rounded-full text-sm font-bold tracking-wide uppercase"
                style={{
                  background: '#FFFFFF',
                  border: '2px solid rgba(252, 211, 77, 0.4)',
                  color: '#D97706',
                  boxShadow: '0 4px 20px rgba(252, 211, 77, 0.25)'
                }}
              >
                💼 Executive Business Impact
              </span>
            </div>

            <h1
              className="text-5xl md:text-7xl lg:text-8xl font-black mb-6"
              style={{
                letterSpacing: '-0.04em',
                lineHeight: 1.05,
                background: 'linear-gradient(135deg, #FCD34D 0%, #F59E0B 100%)',
                WebkitBackgroundClip: 'text',
                WebkitTextFillColor: 'transparent',
                textShadow: '0 0 80px rgba(252, 211, 77, 0.3)'
              }}
            >
              From Video Chaos<br />to Instant Intelligence
            </h1>

            <p
              className="text-xl md:text-2xl lg:text-3xl text-white/90 max-w-4xl mx-auto font-light"
              style={{ lineHeight: 1.5, letterSpacing: '-0.01em' }}
            >
              When semantic search meets high-performance infrastructure,{' '}
              <span className="font-semibold text-amber-300">every frame becomes searchable</span>,{' '}
              <span className="font-semibold text-amber-300">every dataset discoverable</span>,{' '}
              and <span className="font-semibold text-amber-300">every insight actionable</span>.
              <br />
              <span className="text-white/70">AI-powered video analytics and multimodal search that transforms your bottom line.</span>
            </p>
          </motion.div>

          {/* Three Value Pillars */}
          <motion.div
            initial={{ opacity: 0, y: 24 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.6, delay: 0.2, ease: [0.16, 1, 0.3, 1] }}
            className="grid md:grid-cols-3 gap-6 mb-16"
          >
            {/* Cost Reduction Pillar */}
            <div
              className="relative p-8 rounded-2xl overflow-hidden group"
              style={{
                background: 'linear-gradient(135deg, rgba(16, 185, 129, 0.12) 0%, rgba(5, 150, 105, 0.08) 100%)',
                border: '2px solid rgba(16, 185, 129, 0.3)',
                transition: 'all 300ms cubic-bezier(0.16, 1, 0.3, 1)'
              }}
              onMouseEnter={(e) => {
                e.currentTarget.style.transform = 'translateY(-6px)'
                e.currentTarget.style.borderColor = 'rgba(16, 185, 129, 0.6)'
                e.currentTarget.style.boxShadow = '0 20px 60px rgba(16, 185, 129, 0.25)'
              }}
              onMouseLeave={(e) => {
                e.currentTarget.style.transform = 'translateY(0)'
                e.currentTarget.style.borderColor = 'rgba(16, 185, 129, 0.3)'
                e.currentTarget.style.boxShadow = 'none'
              }}
            >
              <div className="mb-4 flex items-center justify-center">
                <div className="p-4 rounded-2xl bg-emerald-500/20">
                  <DollarSign className="w-12 h-12 text-emerald-300" strokeWidth={2.5} />
                </div>
              </div>
              <h3 className="text-2xl md:text-3xl font-bold mb-3 text-emerald-400">
                Cost Reduction
              </h3>
              <div className="space-y-3 text-white/80">
                <div className="flex items-baseline gap-2">
                  <span className="text-lg font-semibold text-emerald-300">Significant storage cost savings</span>
                </div>
                <div className="flex items-baseline gap-2">
                  <span className="text-lg font-semibold text-emerald-300">Improved GPU efficiency</span>
                </div>
                <p className="text-base text-white/60 pt-2 border-t border-white/10">
                  Eliminate cloud egress fees and reduce expensive GPU idle time
                </p>
              </div>
            </div>

            {/* Revenue Acceleration Pillar */}
            <div
              className="relative p-8 rounded-2xl overflow-hidden group"
              style={{
                background: 'linear-gradient(135deg, rgba(252, 211, 77, 0.12) 0%, rgba(245, 158, 11, 0.08) 100%)',
                border: '2px solid rgba(252, 211, 77, 0.3)',
                transition: 'all 300ms cubic-bezier(0.16, 1, 0.3, 1)'
              }}
              onMouseEnter={(e) => {
                e.currentTarget.style.transform = 'translateY(-6px)'
                e.currentTarget.style.borderColor = 'rgba(252, 211, 77, 0.6)'
                e.currentTarget.style.boxShadow = '0 20px 60px rgba(252, 211, 77, 0.25)'
              }}
              onMouseLeave={(e) => {
                e.currentTarget.style.transform = 'translateY(0)'
                e.currentTarget.style.borderColor = 'rgba(252, 211, 77, 0.3)'
                e.currentTarget.style.boxShadow = 'none'
              }}
            >
              <div className="mb-4 flex items-center justify-center">
                <div className="p-4 rounded-2xl bg-amber-500/20">
                  <TrendingUp className="w-12 h-12 text-amber-300" strokeWidth={2.5} />
                </div>
              </div>
              <h3 className="text-2xl md:text-3xl font-bold mb-3 text-amber-400">
                Revenue Acceleration
              </h3>
              <div className="space-y-3 text-white/80">
                <div className="flex items-baseline gap-2">
                  <span className="text-lg font-semibold text-amber-300">Faster time-to-market</span>
                </div>
                <div className="flex items-baseline gap-2">
                  <span className="text-lg font-semibold text-amber-300">Significant productivity improvement</span>
                </div>
                <p className="text-base text-white/60 pt-2 border-t border-white/10">
                  Accelerate product launches with sub-100ms real-time AI experiences
                </p>
              </div>
            </div>

            {/* Competitive Advantage Pillar */}
            <div
              className="relative p-8 rounded-2xl overflow-hidden group"
              style={{
                background: 'linear-gradient(135deg, rgba(237, 39, 56, 0.12) 0%, rgba(220, 38, 38, 0.08) 100%)',
                border: '2px solid rgba(237, 39, 56, 0.3)',
                transition: 'all 300ms cubic-bezier(0.16, 1, 0.3, 1)'
              }}
              onMouseEnter={(e) => {
                e.currentTarget.style.transform = 'translateY(-6px)'
                e.currentTarget.style.borderColor = 'rgba(237, 39, 56, 0.6)'
                e.currentTarget.style.boxShadow = '0 20px 60px rgba(237, 39, 56, 0.25)'
              }}
              onMouseLeave={(e) => {
                e.currentTarget.style.transform = 'translateY(0)'
                e.currentTarget.style.borderColor = 'rgba(237, 39, 56, 0.3)'
                e.currentTarget.style.boxShadow = 'none'
              }}
            >
              <div className="mb-4 flex items-center justify-center">
                <div className="p-4 rounded-2xl bg-red-500/20">
                  <Trophy className="w-12 h-12 text-red-300" strokeWidth={2.5} />
                </div>
              </div>
              <h3 className="text-2xl md:text-3xl font-bold mb-3 text-red-400">
                Competitive Edge
              </h3>
              <div className="space-y-3 text-white/80">
                <div className="flex items-baseline gap-2">
                  <span className="text-lg font-semibold text-red-300">Faster retrieval</span>
                </div>
                <div className="flex items-baseline gap-2">
                  <span className="text-lg font-semibold text-red-300">Infinite scalability</span>
                </div>
                <p className="text-base text-white/60 pt-2 border-t border-white/10">
                  Superior AI performance with model-agnostic flexibility
                </p>
              </div>
            </div>
          </motion.div>

          {/* AI Use Case Highlights */}
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.5, delay: 0.35, ease: [0.16, 1, 0.3, 1] }}
            className="text-center"
          >
            <h3 className="text-2xl md:text-3xl font-bold text-white/90 mb-8">
              Perfect Infrastructure Powers Every AI Initiative
            </h3>

            <div className="grid grid-cols-2 md:grid-cols-4 gap-4 max-w-5xl mx-auto">
              {[
                { icon: <Video className="w-8 h-8 text-purple-400" strokeWidth={2} />, title: 'Video Analytics', desc: 'Real-time frame analysis' },
                { icon: <Search className="w-8 h-8 text-blue-400" strokeWidth={2} />, title: 'Semantic Search', desc: 'Sub-100ms multimodal' },
                { icon: <FolderOpen className="w-8 h-8 text-green-400" strokeWidth={2} />, title: 'Dataset Discovery', desc: 'Instant model experiments' },
                { icon: <Brain className="w-8 h-8 text-pink-400" strokeWidth={2} />, title: 'RAG Applications', desc: 'Knowledge retrieval at scale' }
              ].map((useCase, i) => (
                <div
                  key={i}
                  className="p-6 rounded-xl"
                  style={{
                    background: 'rgba(255, 255, 255, 0.04)',
                    border: '1px solid rgba(255, 255, 255, 0.1)',
                    transition: 'all 200ms ease'
                  }}
                  onMouseEnter={(e) => {
                    e.currentTarget.style.background = 'rgba(255, 255, 255, 0.08)'
                    e.currentTarget.style.borderColor = 'rgba(118, 185, 0, 0.4)'
                  }}
                  onMouseLeave={(e) => {
                    e.currentTarget.style.background = 'rgba(255, 255, 255, 0.04)'
                    e.currentTarget.style.borderColor = 'rgba(255, 255, 255, 0.1)'
                  }}
                >
                  <div className="mb-3 flex justify-center">{useCase.icon}</div>
                  <div className="font-bold text-white text-xl mb-1">{useCase.title}</div>
                  <div className="text-base text-white/60">{useCase.desc}</div>
                </div>
              ))}
            </div>
          </motion.div>

          {/* Bottom Tagline */}
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            transition={{ duration: 0.6, delay: 0.5 }}
            className="text-center mt-16"
          >
            <p
              className="text-xl md:text-2xl lg:text-3xl text-white/50 italic max-w-3xl mx-auto"
              style={{ fontWeight: 300 }}
            >
              "When you can search every video frame, discover every dataset, and surface insights in milliseconds
              <br />you don't just gain efficiency. You unlock competitive intelligence that others take weeks to find."
            </p>
          </motion.div>
        </div>
      </section>

      {/* Hero Section */}
      <section
        className="relative overflow-hidden"
        style={{
          background: 'linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%)'
        }}
      >
        {/* Geometric Background Elements */}
        <div className="absolute inset-0 overflow-hidden pointer-events-none">
          {/* Diagonal Lines Pattern */}
          <div
            className="absolute inset-0 opacity-[0.04]"
            style={{
              backgroundImage: `repeating-linear-gradient(
                -45deg,
                transparent,
                transparent 40px,
                rgba(255,255,255,0.5) 40px,
                rgba(255,255,255,0.5) 41px
              )`
            }}
          />
          {/* Accent Ring - Top Right */}
          <div
            className="absolute"
            style={{
              width: '500px',
              height: '500px',
              border: '1px solid rgba(118, 185, 0, 0.15)',
              borderRadius: '50%',
              right: '-150px',
              top: '-150px',
              animation: 'pulse-ring 8s ease-in-out infinite'
            }}
          />
          <div
            className="absolute"
            style={{
              width: '400px',
              height: '400px',
              border: '1px solid rgba(118, 185, 0, 0.1)',
              borderRadius: '50%',
              right: '-100px',
              top: '-100px',
              animation: 'pulse-ring 8s ease-in-out infinite 0.5s'
            }}
          />
          {/* Accent Ring - Bottom Left */}
          <div
            className="absolute"
            style={{
              width: '400px',
              height: '400px',
              border: '1px solid rgba(237, 39, 56, 0.12)',
              borderRadius: '50%',
              left: '-120px',
              bottom: '-120px',
              animation: 'pulse-ring 8s ease-in-out infinite 1s'
            }}
          />
          {/* Subtle Gradient Overlay */}
          <div
            className="absolute inset-0"
            style={{
              background: 'radial-gradient(ellipse at 30% 20%, rgba(237, 39, 56, 0.08) 0%, transparent 50%), radial-gradient(ellipse at 70% 80%, rgba(118, 185, 0, 0.06) 0%, transparent 50%)'
            }}
          />
        </div>

        <div className="relative z-10 max-w-[1280px] mx-auto px-6 py-20 md:py-28">
          {/* Logo Cards */}
          <motion.div
            initial={{ opacity: 0, y: 24 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.5, ease: [0.16, 1, 0.3, 1] }}
            className="flex items-center justify-center gap-6 md:gap-8 mb-12"
          >
            {/* DDN Card */}
            <div
              className="w-28 h-28 md:w-36 md:h-36 rounded-2xl flex flex-col items-center justify-center"
              style={{
                background: 'linear-gradient(135deg, rgba(237, 39, 56, 0.15) 0%, rgba(237, 39, 56, 0.05) 100%)',
                border: '1px solid rgba(237, 39, 56, 0.2)',
                transition: 'all 200ms cubic-bezier(0.16, 1, 0.3, 1)',
                willChange: 'transform'
              }}
              onMouseEnter={(e) => {
                e.currentTarget.style.transform = 'translateY(-4px) scale(1.02)'
                e.currentTarget.style.borderColor = 'rgba(237, 39, 56, 0.4)'
                e.currentTarget.style.boxShadow = '0 8px 32px rgba(237, 39, 56, 0.2)'
              }}
              onMouseLeave={(e) => {
                e.currentTarget.style.transform = 'translateY(0) scale(1)'
                e.currentTarget.style.borderColor = 'rgba(237, 39, 56, 0.2)'
                e.currentTarget.style.boxShadow = 'none'
              }}
            >
              <img src="/ddn-logo-white.svg" alt="DDN" className="h-10 md:h-12 w-auto mb-2" />
              <span className="text-white/70 text-xs font-medium">INFINIA</span>
            </div>

            {/* Connection Line */}
            <div className="flex items-center gap-2">
              <div className="w-8 h-px bg-gradient-to-r from-ddn-red/50 to-transparent" />
              <div className="w-2 h-2 rounded-full bg-white/30" />
              <div className="w-8 h-px bg-gradient-to-l from-nvidia-green/50 to-transparent" />
            </div>

            {/* NVIDIA Card */}
            <div
              className="w-28 h-28 md:w-36 md:h-36 rounded-2xl flex flex-col items-center justify-center"
              style={{
                background: 'linear-gradient(135deg, rgba(118, 185, 0, 0.15) 0%, rgba(118, 185, 0, 0.05) 100%)',
                border: '1px solid rgba(118, 185, 0, 0.2)',
                transition: 'all 200ms cubic-bezier(0.16, 1, 0.3, 1)',
                willChange: 'transform'
              }}
              onMouseEnter={(e) => {
                e.currentTarget.style.transform = 'translateY(-4px) scale(1.02)'
                e.currentTarget.style.borderColor = 'rgba(118, 185, 0, 0.4)'
                e.currentTarget.style.boxShadow = '0 8px 32px rgba(118, 185, 0, 0.2)'
              }}
              onMouseLeave={(e) => {
                e.currentTarget.style.transform = 'translateY(0) scale(1)'
                e.currentTarget.style.borderColor = 'rgba(118, 185, 0, 0.2)'
                e.currentTarget.style.boxShadow = 'none'
              }}
            >
              <img src="/nvidia-icon.svg" alt="NVIDIA" className="h-12 md:h-14 w-auto" />
              <span className="text-white/70 text-xs font-medium mt-2">NVIDIA</span>
            </div>
          </motion.div>

          {/* Title */}
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.45, delay: 0.08, ease: [0.16, 1, 0.3, 1] }}
            className="text-center mb-6"
          >
            <h1
              className="text-4xl md:text-5xl lg:text-6xl font-bold text-white mb-2"
              style={{ letterSpacing: '-0.03em', lineHeight: 1.1 }}
            >
              <span
                style={{
                  background: 'linear-gradient(90deg, #fff 0%, rgba(255,255,255,0.8) 100%)',
                  WebkitBackgroundClip: 'text',
                  WebkitTextFillColor: 'transparent',
                  backgroundSize: '200% 100%',
                  animation: 'shimmer 3s linear infinite'
                }}
              >
                Multimodal
              </span>{' '}
              <span className="text-white/90">Semantic</span>
            </h1>
            <h2
              className="text-3xl md:text-4xl lg:text-5xl font-bold"
              style={{
                letterSpacing: '-0.03em',
                background: 'linear-gradient(135deg, #ED2738 0%, #76B900 100%)',
                WebkitBackgroundClip: 'text',
                WebkitTextFillColor: 'transparent'
              }}
            >
              Search
            </h2>
          </motion.div>

          {/* Subtitle */}
          <motion.p
            initial={{ opacity: 0, y: 16 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.4, delay: 0.15, ease: [0.16, 1, 0.3, 1] }}
            className="text-center text-white/70 text-lg md:text-xl max-w-2xl mx-auto mb-12"
            style={{ lineHeight: 1.7 }}
          >
            GPU-accelerated AI semantic search across images, videos, and documents.
            <br />
            <span className="text-nvidia-green font-semibold">Powered by NVIDIA GPU</span> + DDN INFINIA Storage
          </motion.p>

          {/* Stats Grid */}
          <motion.div
            initial={{ opacity: 0, y: 16 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.4, delay: 0.22, ease: [0.16, 1, 0.3, 1] }}
            className="grid grid-cols-2 md:grid-cols-4 gap-4 max-w-4xl mx-auto"
          >
            <StatCard value="CLIP" label="Image Embeddings" description="OpenAI Vision" />
            <StatCard value="BLIP" label="Image Captioning" description="Salesforce AI" />
            <StatCard value="<50ms" label="Search Latency" description="Semantic matching" />
            <StatCard value="3" label="Modalities" description="Image, Video, Doc" />
          </motion.div>
        </div>
      </section>

      {/* Features Section */}
      <section className="bg-surface-primary px-6 py-16">
        <div className="max-w-[1280px] mx-auto">
          <div className="text-center mb-10">
            <span className="eyebrow text-nvidia-green">Capabilities</span>
            <h2 className="heading-2 mt-2 mb-3">
              Multimodal AI Search
            </h2>
            <p className="body-text max-w-2xl mx-auto">
              Search across different content types using natural language. Our AI understands the meaning behind your queries.
            </p>
          </div>

          {/* Feature Cards */}
          <div className="grid md:grid-cols-3 gap-6 mb-12">
            <FeatureCard
              icon={<Image className="w-8 h-8" />}
              title="Image Search"
              description="Upload images and search using natural language. CLIP embeddings enable semantic understanding of visual content."
              features={['Auto-captioning with BLIP', 'Object detection', 'Semantic embeddings']}
              color="ddn"
            />
            <FeatureCard
              icon={<Video className="w-8 h-8" />}
              title="Video Search"
              description="Process videos frame-by-frame for comprehensive understanding. Find moments based on visual or textual descriptions."
              features={['Frame extraction', 'Scene analysis', 'Motion detection']}
              color="nvidia"
            />
            <FeatureCard
              icon={<FileText className="w-8 h-8" />}
              title="Document Search"
              description="Semantic search across PDF and Word documents. AI summarization extracts key insights automatically."
              features={['Text extraction', 'Auto summarization', 'Key term identification']}
              color="blue"
            />
          </div>

          {/* How It Works */}
          <div className="mt-16">
            <h3 className="heading-3 mb-8 text-center">How It Works</h3>
            <div className="grid md:grid-cols-4 gap-4">
              <StepCard number="01" title="Upload" description="Upload images, videos, or documents to DDN INFINIA storage" />
              <StepCard number="02" title="Analyze" description="AI models analyze and extract semantic meaning from content" />
              <StepCard number="03" title="Embed" description="Generate embeddings and export to JSON, stored in DDN INFINIA" />
              <StepCard number="04" title="Search" description="Query using natural language and find relevant content" />
            </div>
          </div>

          {/* Architecture Diagram */}
          <div className="mt-16">
            <h3 className="heading-3 mb-8 text-center">Architecture Diagram</h3>
            <div className="card p-8" style={{ background: 'var(--surface-card)' }}>
              {/* Diagram Container */}
              <div className="flex items-center justify-center gap-4 flex-wrap md:flex-nowrap">
                {/* User */}
                <div className="flex flex-col items-center gap-2">
                  <div className="w-16 h-16 rounded-full border-2 flex items-center justify-center" style={{ borderColor: '#007AFF', color: '#007AFF' }}>
                    <svg className="w-8 h-8" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M16 7a4 4 0 11-8 0 4 4 0 018 0zM12 14a7 7 0 00-7 7h14a7 7 0 00-7-7z" />
                    </svg>
                  </div>
                  <span className="text-xs font-medium" style={{ color: 'var(--text-muted)' }}>User</span>
                </div>

                {/* Arrow 1 */}
                <div className="flex flex-col items-center">
                  <svg className="w-8 h-6" viewBox="0 0 32 24" fill="none">
                    <path d="M0 12 L24 12 M18 6 L24 12 L18 18" stroke="#8E8E93" strokeWidth="2" fill="none" />
                  </svg>
                  <span className="text-[10px]" style={{ color: 'var(--text-muted)' }}>HTTP</span>
                </div>

                {/* React Frontend */}
                <div className="flex flex-col items-center">
                  <div className="px-6 py-4 rounded-xl border-2 min-w-[140px] text-center" style={{ background: 'rgba(0, 122, 255, 0.05)', borderColor: 'rgba(0, 122, 255, 0.3)' }}>
                    <div className="font-semibold text-sm mb-1" style={{ color: '#007AFF' }}>React Web App</div>
                    <div className="text-xs" style={{ color: 'var(--text-muted)' }}>TypeScript + Vite</div>
                  </div>
                  <span className="text-[10px] mt-1" style={{ color: '#007AFF' }}>Frontend</span>
                </div>

                {/* Arrow 2 */}
                <div className="flex flex-col items-center">
                  <svg className="w-8 h-6" viewBox="0 0 32 24" fill="none">
                    <path d="M0 12 L24 12 M18 6 L24 12 L18 18" stroke="#8E8E93" strokeWidth="2" fill="none" />
                  </svg>
                  <span className="text-[10px]" style={{ color: 'var(--text-muted)' }}>API</span>
                </div>

                {/* FastAPI Backend */}
                <div className="flex flex-col items-center">
                  <div className="px-6 py-4 rounded-xl border-2 min-w-[140px] text-center" style={{ background: 'rgba(255, 149, 0, 0.05)', borderColor: 'rgba(255, 149, 0, 0.3)' }}>
                    <div className="font-semibold text-sm mb-1" style={{ color: '#FF9500' }}>FastAPI Server</div>
                    <div className="text-xs" style={{ color: 'var(--text-muted)' }}>Python REST API</div>
                  </div>
                  <span className="text-[10px] mt-1" style={{ color: '#FF9500' }}>Backend</span>
                </div>

                {/* Arrow 3 */}
                <div className="flex flex-col items-center">
                  <svg className="w-8 h-6" viewBox="0 0 32 24" fill="none">
                    <path d="M0 12 L24 12 M18 6 L24 12 L18 18" stroke="#8E8E93" strokeWidth="2" fill="none" />
                  </svg>
                  <span className="text-[10px]" style={{ color: 'var(--text-muted)' }}>Inference</span>
                </div>

                {/* GPU AI */}
                <div className="flex flex-col items-center">
                  <div className="px-6 py-4 rounded-xl border-2 min-w-[180px] text-center" style={{ background: 'rgba(118, 185, 0, 0.05)', borderColor: 'rgba(118, 185, 0, 0.3)' }}>
                    <div className="font-semibold text-sm mb-1 flex items-center justify-center gap-1" style={{ color: '#76B900' }}>
                      <Zap className="w-3.5 h-3.5" />
                      GPU-Accelerated AI
                    </div>
                    <div className="flex gap-2 justify-center my-2">
                      <span className="px-2 py-0.5 rounded text-xs bg-white/50" style={{ color: '#76B900' }}>CLIP</span>
                      <span className="px-2 py-0.5 rounded text-xs bg-white/50" style={{ color: '#76B900' }}>BLIP</span>
                      <span className="px-2 py-0.5 rounded text-xs bg-white/50" style={{ color: '#76B900' }}>ViT</span>
                    </div>
                    <div className="text-xs" style={{ color: 'var(--text-muted)' }}>CUDA PyTorch</div>
                  </div>
                  <span className="text-[10px] mt-1" style={{ color: '#76B900' }}>AI Models</span>
                </div>

                {/* Arrow 4 */}
                <div className="flex flex-col items-center">
                  <svg className="w-8 h-6" viewBox="0 0 32 24" fill="none">
                    <path d="M0 12 L24 12 M18 6 L24 12 L18 18" stroke="#8E8E93" strokeWidth="2" fill="none" />
                  </svg>
                  <span className="text-[10px]" style={{ color: 'var(--text-muted)' }}>S3</span>
                </div>

                {/* DDN Storage */}
                <div className="flex flex-col items-center">
                  <div className="px-6 py-4 rounded-xl border-2 min-w-[140px] text-center" style={{ background: 'rgba(237, 39, 56, 0.05)', borderColor: 'rgba(237, 39, 56, 0.3)' }}>
                    <div className="font-semibold text-sm mb-1" style={{ color: '#ED2738' }}>DDN INFINIA</div>
                    <div className="text-xs" style={{ color: 'var(--text-muted)' }}>Object Storage</div>
                  </div>
                  <span className="text-[10px] mt-1" style={{ color: '#ED2738' }}>Storage</span>
                </div>
              </div>

              {/* Data Types Row */}
              <div className="flex items-center justify-center gap-6 mt-8 pt-6 border-t" style={{ borderColor: 'var(--border-subtle)' }}>
                <div className="flex items-center gap-2">
                  <Image className="w-4 h-4" style={{ color: '#9333EA' }} />
                  <span className="text-xs font-semibold" style={{ color: '#9333EA' }}>Images</span>
                </div>
                <div className="flex items-center gap-2">
                  <Video className="w-4 h-4" style={{ color: '#DC2626' }} />
                  <span className="text-xs font-semibold" style={{ color: '#DC2626' }}>Videos</span>
                </div>
                <div className="flex items-center gap-2">
                  <FileText className="w-4 h-4" style={{ color: '#2563EB' }} />
                  <span className="text-xs font-semibold" style={{ color: '#2563EB' }}>Documents</span>
                </div>
              </div>

              <p className="text-center text-sm mt-6" style={{ color: 'var(--text-muted)' }}>
                End-to-end architecture showing data flow from user interface through GPU-accelerated AI processing to DDN INFINIA storage
              </p>
            </div>
          </div>
        </div>
      </section>

      {/* Technology Section */}
      <section className="bg-surface-base px-6 py-16">
        <div className="max-w-[1280px] mx-auto">
          <div className="text-center mb-10">
            <span className="eyebrow text-ddn-red">Technology</span>
            <h2 className="heading-2 mt-2 mb-3">
              Powered By
            </h2>
            <p className="body-text max-w-2xl mx-auto mb-8">
              Built on enterprise-grade infrastructure with NVIDIA GPU-accelerated AI models for maximum performance.
            </p>

            {/* Logo Grid */}
            <div className="flex items-center justify-center gap-12 mb-12">
              <div className="flex flex-col items-center">
                <img src="/logo-ddn.svg" alt="DDN" className="h-12 w-auto" />
              </div>
              <div className="text-4xl text-text-muted">×</div>
              <div className="flex flex-col items-center">
                <img src="/nvidia-logo-light.png" alt="NVIDIA" className="h-12 w-auto" />
              </div>
            </div>
          </div>

          {/* GPU Acceleration Highlight */}
          <div className="card p-6 mb-8 border-2" style={{ borderColor: 'rgba(118, 185, 0, 0.3)', background: 'linear-gradient(135deg, rgba(118, 185, 0, 0.05) 0%, rgba(118, 185, 0, 0.02) 100%)' }}>
            <div className="flex items-start gap-4">
              <div className="p-3 rounded-xl bg-nvidia-green/10">
                <Zap className="w-8 h-8 text-nvidia-green" />
              </div>
              <div>
                <h3 className="heading-3 mb-2 flex items-center gap-2">
                  <span>NVIDIA GPU Acceleration</span>
                  <span className="px-2 py-0.5 rounded-md bg-nvidia-green/10 text-nvidia-green text-xs font-bold">POWERED</span>
                </h3>
                <p className="body-text mb-4">
                  All AI models run on NVIDIA GPUs for blazing-fast inference. CUDA-optimized PyTorch delivers up to <strong>100x faster</strong> processing compared to CPU-only inference, enabling real-time semantic search across massive media libraries.
                </p>
                <div className="grid md:grid-cols-3 gap-4">
                  <div className="flex items-start gap-2">
                    <Zap className="w-5 h-5 text-nvidia-green" />
                    <div>
                      <div className="text-sm font-semibold text-text-primary">Instant Embeddings</div>
                      <div className="text-xs text-text-muted">CLIP/BLIP on GPU</div>
                    </div>
                  </div>
                  <div className="flex items-start gap-2">
                    <svg className="w-5 h-5 text-nvidia-green" fill="currentColor" viewBox="0 0 20 20">
                      <path d="M10 2L12 8H18L13 12L15 18L10 14L5 18L7 12L2 8H8L10 2Z" />
                    </svg>
                    <div>
                      <div className="text-sm font-semibold text-text-primary">Batch Processing</div>
                      <div className="text-xs text-text-muted">Parallel workloads</div>
                    </div>
                  </div>
                  <div className="flex items-start gap-2">
                    <svg className="w-5 h-5 text-nvidia-green" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 7h8m0 0v8m0-8l-8 8-4-4-6 6" />
                    </svg>
                    <div>
                      <div className="text-sm font-semibold text-text-primary">Auto Scaling</div>
                      <div className="text-xs text-text-muted">CPU fallback ready</div>
                    </div>
                  </div>
                </div>
              </div>
            </div>
          </div>

          <div className="grid md:grid-cols-2 gap-6 mb-12">
            <div className="card p-6">
              <h3 className="heading-3 mb-4 flex items-center gap-2">
                <Database className="w-5 h-5 text-ddn-red" />
                Storage & Infrastructure
              </h3>
              <ul className="space-y-3">
                {[
                  { name: 'DDN INFINIA', desc: 'High-performance S3-compatible storage with GPU-Direct capabilities' },
                  { name: 'FastAPI', desc: 'Modern async Python framework optimized for GPU workloads' },
                  { name: 'PyTorch', desc: 'CUDA-accelerated deep learning framework for GPU inference' },
                ].map((tech) => (
                  <li key={tech.name} className="flex items-start gap-3 text-sm">
                    <span className="w-1.5 h-1.5 bg-ddn-red rounded-full mt-2 flex-shrink-0" />
                    <div>
                      <span className="font-medium" style={{ color: 'var(--text-primary)' }}>{tech.name}</span>
                      <span style={{ color: 'var(--text-muted)' }}> — {tech.desc}</span>
                    </div>
                  </li>
                ))}
              </ul>
            </div>

            <div className="card p-6">
              <h3 className="heading-3 mb-4 flex items-center gap-2">
                <Zap className="w-5 h-5 text-nvidia-green" />
                GPU-Accelerated AI Models
              </h3>
              <ul className="space-y-3">
                {[
                  { name: 'CLIP', desc: 'Vision-language embeddings with NVIDIA tensor cores' },
                  { name: 'BLIP', desc: 'GPU-accelerated image captioning by Salesforce' },
                  { name: 'ViT', desc: 'Vision transformer for scene classification on GPU' },
                ].map((tech) => (
                  <li key={tech.name} className="flex items-start gap-3 text-sm">
                    <span className="w-1.5 h-1.5 bg-nvidia-green rounded-full mt-2 flex-shrink-0" />
                    <div>
                      <span className="font-medium" style={{ color: 'var(--text-primary)' }}>{tech.name}</span>
                      <span style={{ color: 'var(--text-muted)' }}> — {tech.desc}</span>
                    </div>
                  </li>
                ))}
              </ul>
            </div>
          </div>

          {/* Use Cases */}
          <div className="mb-12">
            <h3 className="heading-3 mb-6 text-center">Use Cases</h3>
            <div className="flex flex-wrap justify-center gap-2">
              {[
                'Digital Asset Management',
                'Media Libraries',
                'Content Discovery',
                'Visual Search',
                'Document Intelligence',
                'Video Analysis',
                'E-commerce',
                'Brand Monitoring'
              ].map((useCase) => (
                <span
                  key={useCase}
                  className="bg-surface-secondary px-4 py-2 rounded-full text-sm font-medium border transition-colors"
                  style={{
                    color: 'var(--text-secondary)',
                    borderColor: 'var(--border-subtle)'
                  }}
                >
                  {useCase}
                </span>
              ))}
            </div>
          </div>
        </div>
      </section>

      {/* CTA Section */}
      <section className="px-6 py-20" style={{ background: 'linear-gradient(135deg, var(--ddn-red) 0%, var(--ddn-red-hover) 100%)' }}>
        <div className="max-w-3xl mx-auto text-center">
          <motion.div
            initial={{ opacity: 0, y: 16 }}
            whileInView={{ opacity: 1, y: 0 }}
            viewport={{ once: true, margin: '-50px' }}
            transition={{ duration: 0.4, ease: [0.16, 1, 0.3, 1] }}
          >
            <h2 className="text-3xl md:text-4xl font-bold text-white mb-4" style={{ letterSpacing: '-0.02em' }}>
              Ready to Try It?
            </h2>
            <p className="text-white/80 text-lg mb-8 max-w-xl mx-auto">
              Configure your storage, upload content, and experience AI-powered multimodal search.
            </p>
            <button
              onClick={onStartDemo}
              className="inline-flex items-center gap-3 bg-white text-neutral-900 px-8 py-4 rounded-xl font-semibold text-lg"
              style={{
                transition: 'transform 200ms cubic-bezier(0.16, 1, 0.3, 1), box-shadow 200ms cubic-bezier(0.16, 1, 0.3, 1)',
                willChange: 'transform'
              }}
              onMouseEnter={(e) => {
                e.currentTarget.style.transform = 'scale(1.03) translateZ(0)'
                e.currentTarget.style.boxShadow = '0 20px 40px rgba(0,0,0,0.2)'
              }}
              onMouseLeave={(e) => {
                e.currentTarget.style.transform = 'scale(1) translateZ(0)'
                e.currentTarget.style.boxShadow = 'none'
              }}
              onMouseDown={(e) => {
                e.currentTarget.style.transform = 'scale(0.98) translateZ(0)'
              }}
              onMouseUp={(e) => {
                e.currentTarget.style.transform = 'scale(1.03) translateZ(0)'
              }}
            >
              Start the Demo
              <ArrowRight className="w-5 h-5" />
            </button>
          </motion.div>
        </div>
      </section>

      {/* Footer */}
      <section className="bg-neutral-900 px-6 py-8">
        <div className="max-w-[1280px] mx-auto text-center">
          <p className="text-white/40 text-sm">
            DDN INFINIA Multimodal Semantic Search Demo
          </p>
        </div>
      </section>
    </div>
  )
}

function StatCard({ value, label, description }: { value: string; label: string; description: string }) {
  return (
    <div
      className="p-5 rounded-xl text-center relative overflow-hidden group"
      style={{
        background: 'rgba(255, 255, 255, 0.05)',
        border: '1px solid rgba(255, 255, 255, 0.1)',
        transition: 'all 200ms ease'
      }}
      onMouseEnter={(e) => {
        e.currentTarget.style.background = 'rgba(255, 255, 255, 0.08)'
        e.currentTarget.style.borderColor = 'rgba(255, 255, 255, 0.15)'
      }}
      onMouseLeave={(e) => {
        e.currentTarget.style.background = 'rgba(255, 255, 255, 0.05)'
        e.currentTarget.style.borderColor = 'rgba(255, 255, 255, 0.1)'
      }}
    >
      <div
        className="text-2xl md:text-3xl font-bold font-mono mb-1 text-white"
      >
        {value}
      </div>
      <div className="text-white/80 font-medium text-sm">{label}</div>
      <div className="text-white/40 text-xs mt-1">{description}</div>
    </div>
  )
}

function FeatureCard({
  icon,
  title,
  description,
  features,
  color
}: {
  icon: React.ReactNode
  title: string
  description: string
  features: string[]
  color: 'ddn' | 'nvidia' | 'blue'
}) {
  const colors = {
    ddn: { hex: '#ED2738', bg: 'rgba(237, 39, 56, 0.08)' },
    nvidia: { hex: '#76B900', bg: 'rgba(118, 185, 0, 0.08)' },
    blue: { hex: '#1A81AF', bg: 'rgba(26, 129, 175, 0.08)' }
  }

  return (
    <div className="card p-6 relative overflow-hidden">
      <div
        className="absolute top-0 left-0 w-full h-1"
        style={{ backgroundColor: colors[color].hex }}
      />
      <div
        className="w-14 h-14 rounded-xl flex items-center justify-center mb-4"
        style={{ backgroundColor: colors[color].bg, color: colors[color].hex }}
      >
        {icon}
      </div>
      <h4 className="font-semibold text-lg mb-2" style={{ color: 'var(--text-primary)' }}>{title}</h4>
      <p className="text-sm mb-4" style={{ color: 'var(--text-secondary)' }}>{description}</p>
      <ul className="space-y-2">
        {features.map((feature, i) => (
          <li key={i} className="flex items-center gap-2 text-sm" style={{ color: 'var(--text-muted)' }}>
            <span
              className="w-1.5 h-1.5 rounded-full flex-shrink-0"
              style={{ backgroundColor: colors[color].hex }}
            />
            {feature}
          </li>
        ))}
      </ul>
    </div>
  )
}

function StepCard({ number, title, description }: { number: string; title: string; description: string }) {
  return (
    <div className="card p-5 text-center">
      <div
        className="inline-flex items-center justify-center w-10 h-10 rounded-full text-sm font-bold mb-3"
        style={{ backgroundColor: 'var(--ddn-red-light)', color: 'var(--ddn-red)' }}
      >
        {number}
      </div>
      <h4 className="font-semibold mb-2" style={{ color: 'var(--text-primary)' }}>{title}</h4>
      <p className="text-sm" style={{ color: 'var(--text-muted)' }}>{description}</p>
    </div>
  )
}
