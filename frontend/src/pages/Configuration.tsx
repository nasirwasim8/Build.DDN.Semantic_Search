import { useState, useEffect } from 'react'
import { useMutation } from '@tanstack/react-query'
import { CheckCircle, XCircle, Loader2, Server, FolderOpen, Info } from 'lucide-react'
import toast from 'react-hot-toast'
import { api, StorageConfig } from '../services/api'

export default function ConfigurationPage() {
  const [ddnConfig, setDdnConfig] = useState<StorageConfig>({
    access_key: '',
    secret_key: '',
    bucket_name: '',
    endpoint_url: '',
    region: 'us-east-1',
  })

  const [localCacheConfig, setLocalCacheConfig] = useState({
    enabled: false,
    videos_path: '',
    embeddings_path: '',
  })

  const [ddnStatus, setDdnStatus] = useState<{ connected: boolean; latency?: number } | null>(null)
  const [configLoaded, setConfigLoaded] = useState(false)

  // Load saved configuration on page mount
  useEffect(() => {
    const loadSavedConfig = async () => {
      try {
        const data = await api.loadConfig()
        if (data.success && data.ddn_config) {
          const { ddn_config } = data
          // Only load if there are actual saved credentials
          if (ddn_config.access_key || ddn_config.bucket_name) {
            setDdnConfig({
              access_key: ddn_config.access_key || '',
              secret_key: ddn_config.secret_key || '',
              bucket_name: ddn_config.bucket_name || '',
              endpoint_url: ddn_config.endpoint_url || '',
              region: ddn_config.region || 'us-east-1',
            })
            setConfigLoaded(true)
          }
        }

        // Load local cache config
        if (data.local_cache_config) {
          setLocalCacheConfig({
            enabled: data.local_cache_config.enabled || false,
            videos_path: data.local_cache_config.videos_path || '',
            embeddings_path: data.local_cache_config.embeddings_path || '',
          })
        }
      } catch (error) {
        console.error('Failed to load saved config:', error)
      }
    }
    loadSavedConfig()
  }, [])

  const saveDdnMutation = useMutation({
    mutationFn: () => api.configureDDN({ ...ddnConfig, endpoint_url: ddnConfig.endpoint_url || '' }),
    onSuccess: () => {
      toast.success('DDN INFINIA configuration saved')
    },
    onError: () => {
      toast.error('Failed to save DDN configuration')
    },
  })

  const saveLocalCacheMutation = useMutation({
    mutationFn: () => api.configureLocalCache(localCacheConfig),
    onSuccess: () => {
      toast.success('Local cache configuration saved')
    },
    onError: (error: any) => {
      toast.error(error.response?.data?.detail || 'Failed to save local cache configuration')
    },
  })

  const testDdnMutation = useMutation({
    mutationFn: () => api.testConnection('ddn_infinia'),
    onSuccess: (data) => {
      setDdnStatus({ connected: data.success, latency: data.latency_ms })
      if (data.success) {
        toast.success('DDN INFINIA connection successful')
      } else {
        toast.error(data.message || 'DDN INFINIA connection failed')
      }
    },
  })

  return (
    <div className="space-y-8">
      {/* Section Header */}
      <div className="section-header">
        <h2 className="section-title">Storage Configuration</h2>
        <p className="section-description">
          Configure your DDN INFINIA storage for multimodal content storage and retrieval.
        </p>
      </div>

      {/* Connection Status Bar */}
      <div className="toolbar justify-between">
        <div className="flex items-center gap-3">
          <div className={`status-dot ${ddnStatus?.connected ? 'status-dot-success status-dot-pulse' : 'status-dot-error'}`} />
          <div>
            <span className="text-sm font-medium text-neutral-900">DDN INFINIA</span>
            {ddnStatus?.latency && (
              <span className="text-xs text-neutral-500 ml-2">{ddnStatus.latency.toFixed(0)}ms</span>
            )}
          </div>
        </div>

        <button
          onClick={() => testDdnMutation.mutate()}
          className="btn-primary"
          disabled={testDdnMutation.isPending}
        >
          {testDdnMutation.isPending ? (
            <Loader2 className="w-4 h-4 animate-spin" />
          ) : (
            'Test Connection'
          )}
        </button>
      </div>

      {/* DDN INFINIA Configuration */}
      <div className="card-elevated p-6 card-accent-ddn">
        {/* Header */}
        <div className="flex items-center justify-between mb-6">
          <div className="flex items-center gap-3">
            <div className="w-10 h-10 rounded-xl flex items-center justify-center bg-ddn-red/10">
              <Server className="w-5 h-5 text-ddn-red" />
            </div>
            <div>
              <h3 className="font-semibold text-neutral-900">DDN INFINIA</h3>
              <p className="text-xs text-neutral-500">High-performance S3-compatible storage</p>
            </div>
          </div>
          {ddnStatus?.connected && (
            <div className="badge badge-success">
              <CheckCircle className="w-3.5 h-3.5" />
              Connected
            </div>
          )}
          {ddnStatus !== null && !ddnStatus?.connected && (
            <div className="badge badge-error">
              <XCircle className="w-3.5 h-3.5" />
              Failed
            </div>
          )}
        </div>

        {/* Config Loaded Indicator */}
        {configLoaded && (
          <div className="mb-4 flex items-center gap-2 p-2 rounded-lg bg-status-info-subtle text-status-info text-sm">
            <CheckCircle className="w-4 h-4" />
            Configuration loaded from saved settings
          </div>
        )}

        {/* Form Fields */}
        <div className="grid md:grid-cols-2 gap-4">
          <FormField
            label="Access Key"
            type="password"
            value={ddnConfig.access_key}
            onChange={(v) => setDdnConfig({ ...ddnConfig, access_key: v })}
            placeholder="Enter access key"
          />
          <FormField
            label="Secret Key"
            type="password"
            value={ddnConfig.secret_key}
            onChange={(v) => setDdnConfig({ ...ddnConfig, secret_key: v })}
            placeholder="Enter secret key"
          />
          <FormField
            label="Bucket Name"
            value={ddnConfig.bucket_name}
            onChange={(v) => setDdnConfig({ ...ddnConfig, bucket_name: v })}
            placeholder="my-multimodal-bucket"
          />
          <FormField
            label="Region"
            value={ddnConfig.region}
            onChange={(v) => setDdnConfig({ ...ddnConfig, region: v })}
            placeholder="us-east-1"
          />
          <div className="md:col-span-2">
            <FormField
              label="Endpoint URL"
              value={ddnConfig.endpoint_url || ''}
              onChange={(v) => setDdnConfig({ ...ddnConfig, endpoint_url: v })}
              placeholder="https://your-ddn-infinia-endpoint.com"
            />
          </div>
        </div>

        {/* Actions */}
        <div className="flex gap-3 mt-6 pt-6 border-t border-neutral-100">
          <button
            onClick={() => saveDdnMutation.mutate()}
            disabled={saveDdnMutation.isPending}
            className="btn-primary"
          >
            {saveDdnMutation.isPending ? <Loader2 className="w-4 h-4 animate-spin" /> : 'Save Configuration'}
          </button>
          <button
            onClick={() => testDdnMutation.mutate()}
            disabled={testDdnMutation.isPending}
            className="btn-secondary"
          >
            {testDdnMutation.isPending ? <Loader2 className="w-4 h-4 animate-spin" /> : 'Test Connection'}
          </button>
        </div>

        {/* Status Message */}
        {ddnStatus !== null && (
          <div className={`mt-4 flex items-center gap-2 p-3 rounded-lg text-sm font-medium ${ddnStatus.connected
            ? 'status-banner-success'
            : 'bg-status-error-subtle text-status-error border border-status-error/20'
            }`}>
            {ddnStatus.connected ? (
              <>
                <CheckCircle className="w-4 h-4" />
                Connection verified
                {ddnStatus.latency && <span className="ml-auto text-xs opacity-70">{ddnStatus.latency.toFixed(1)}ms latency</span>}
              </>
            ) : (
              <>
                <XCircle className="w-4 h-4" />
                Connection failed — check credentials and endpoint
              </>
            )}
          </div>
        )}
      </div>

      {/* Local Cache Configuration */}
      <div className="card-elevated p-6" style={{ borderLeft: '3px solid #F59E0B' }}>
        <div className="flex items-center justify-between mb-6">
          <div className="flex items-center gap-3">
            <div className="w-10 h-10 rounded-xl flex items-center justify-center bg-amber-500/10">
              <FolderOpen className="w-5 h-5 text-amber-600" />
            </div>
            <div>
              <h3 className="font-semibold text-neutral-900">Local Cache</h3>
              <p className="text-xs text-neutral-500">Configure local file caching for demo mode</p>
            </div>
          </div>
          {localCacheConfig.enabled && (
            <div className="badge" style={{ backgroundColor: '#FEF3C7', color: '#92400E', border: '1px solid #FCD34D' }}>
              <CheckCircle className="w-3.5 h-3.5" />
              Enabled
            </div>
          )}
        </div>

        {/* Info Alert */}
        <div className="mb-6 flex items-start gap-3 p-4 rounded-lg bg-amber-50 border border-amber-200">
          <Info className="w-5 h-5 text-amber-600 flex-shrink-0 mt-0.5" />
          <div className="text-sm text-amber-800">
            <p className="font-medium mb-1">Demo Mode</p>
            <p>When enabled, the application will ONLY use local files for search and playback. No S3 calls will be made. Perfect for offline demos or conferences with unreliable internet.</p>
          </div>
        </div>

        {/* Enable Toggle */}
        <div className="mb-6 flex items-center justify-between p-4 rounded-lg bg-neutral-50">
          <div>
            <label className="text-sm font-medium text-neutral-900">Enable Local Cache Mode</label>
            <p className="text-xs text-neutral-500 mt-1">Use local files instead of S3 storage</p>
          </div>
          <button
            onClick={() => setLocalCacheConfig({ ...localCacheConfig, enabled: !localCacheConfig.enabled })}
            className={`relative inline-flex h-6 w-11 items-center rounded-full transition-colors ${localCacheConfig.enabled ? 'bg-amber-500' : 'bg-neutral-300'
              }`}
          >
            <span
              className={`inline-block h-4 w-4 transform rounded-full bg-white transition-transform ${localCacheConfig.enabled ? 'translate-x-6' : 'translate-x-1'
                }`}
            />
          </button>
        </div>

        {/* Path Fields */}
        <div className="space-y-4">
          <FormField
            label="Videos Directory Path"
            value={localCacheConfig.videos_path}
            onChange={(v) => setLocalCacheConfig({ ...localCacheConfig, videos_path: v })}
            placeholder="/path/to/cache/videos"
            disabled={!localCacheConfig.enabled}
          />
          <FormField
            label="Embeddings Directory Path"
            value={localCacheConfig.embeddings_path}
            onChange={(v) => setLocalCacheConfig({ ...localCacheConfig, embeddings_path: v })}
            placeholder="/path/to/cache/embeddings"
            disabled={!localCacheConfig.enabled}
          />
        </div>

        {/* Actions */}
        <div className="flex gap-3 mt-6 pt-6 border-t border-neutral-100">
          <button
            onClick={() => saveLocalCacheMutation.mutate()}
            disabled={saveLocalCacheMutation.isPending}
            className="btn-primary"
            style={{ backgroundColor: '#F59E0B', borderColor: '#F59E0B' }}
          >
            {saveLocalCacheMutation.isPending ? <Loader2 className="w-4 h-4 animate-spin" /> : 'Save Local Cache Configuration'}
          </button>
        </div>
      </div>

      {/* Info Section */}
      <div className="card p-6">
        <h3 className="font-semibold mb-4" style={{ color: 'var(--text-primary)' }}>Configuration Notes</h3>
        <ul className="space-y-2 text-sm" style={{ color: 'var(--text-secondary)' }}>
          <li className="flex items-start gap-2">
            <span className="w-1.5 h-1.5 bg-ddn-red rounded-full mt-2 flex-shrink-0" />
            DDN INFINIA uses S3-compatible API with self-signed certificates
          </li>
          <li className="flex items-start gap-2">
            <span className="w-1.5 h-1.5 bg-ddn-red rounded-full mt-2 flex-shrink-0" />
            Ensure your bucket exists and has appropriate permissions
          </li>
          <li className="flex items-start gap-2">
            <span className="w-1.5 h-1.5 bg-ddn-red rounded-full mt-2 flex-shrink-0" />
            Content will be organized by modality (images/, videos/, documents/)
          </li>
        </ul>
      </div>
    </div>
  )
}

interface FormFieldProps {
  label: string
  value: string
  onChange: (value: string) => void
  placeholder?: string
  type?: string
  disabled?: boolean
}

function FormField({ label, value, onChange, placeholder, type = 'text', disabled = false }: FormFieldProps) {
  return (
    <div>
      <label className="block text-xs font-medium text-neutral-500 uppercase tracking-wide mb-2">
        {label}
      </label>
      <input
        type={type}
        value={value}
        onChange={(e) => onChange(e.target.value)}
        placeholder={placeholder}
        disabled={disabled}
        className="input-field"
      />
    </div>
  )
}
