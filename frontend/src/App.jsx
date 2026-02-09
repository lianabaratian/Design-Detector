import { useState } from 'react';
import axios from 'axios';
import { ReactCompareSlider, ReactCompareSliderImage } from 'react-compare-slider';
import { Upload, Eye, Activity } from 'lucide-react';

const API_URL = 'http://localhost:8000/predict';

function getRecommendation(score) {
  if (score < 0.5) return 'Increase contrast and clarity for better attention.';
  if (score < 0.8) return 'Good, but consider improving focal points.';
  return 'Excellent! Your design captures attention well.';
}

function App() {
  const [uploadedImage, setUploadedImage] = useState(null);
  const [heatmapBase64, setHeatmapBase64] = useState('');
  const [attentionScore, setAttentionScore] = useState(null);
  const [loading, setLoading] = useState(false);

  const handleDrop = async (e) => {
    e.preventDefault();
    const file = e.dataTransfer.files[0];
    if (file) {
      uploadImage(file);
    }
  };

  const handleFileChange = (e) => {
    const file = e.target.files[0];
    if (file) {
      uploadImage(file);
    }
  };

  const uploadImage = async (file) => {
    setLoading(true);
    setUploadedImage(URL.createObjectURL(file));
    const formData = new FormData();
    formData.append('file', file);

    try {
      const res = await axios.post(API_URL, formData, {
        headers: { 'Content-Type': 'multipart/form-data' },
      });
      setHeatmapBase64(res.data.heatmap);
      setAttentionScore(res.data.attention_score / 100);
    } catch (err) {
      console.error('Error uploading image:', err);
      setHeatmapBase64('');
      setAttentionScore(null);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="min-h-screen flex flex-col" style={{ backgroundColor: '#0a0a12' }}>
      {/* Header */}
      <header className="flex items-center justify-between px-8 py-6 border-b border-slate-800">
        <div className="flex items-center gap-3">
          <Eye className="w-7 h-7" style={{ color: '#00eaff' }} />
          <h1 className="text-2xl font-bold tracking-wide text-white">VisionUX AI</h1>
        </div>
        <span className="px-3 py-1 rounded-full text-xs font-semibold" style={{ backgroundColor: '#00eaff', color: '#0a0a12' }}>
          AI-Powered UX Audit
        </span>
      </header>

      {/* Main Content */}
      <main className="flex flex-col md:flex-row flex-1">
        {/* Left: Upload & Compare */}
        <section className="flex-1 flex flex-col items-center justify-center p-8">
          {/* Upload Zone */}
          <div
            className="w-full max-w-md h-56 border-2 border-dashed border-electric-blue rounded-xl flex flex-col items-center justify-center cursor-pointer bg-slate-900 hover:bg-slate-800 transition mb-8"
            onDrop={handleDrop}
            onDragOver={(e) => e.preventDefault()}
            onClick={() => document.getElementById('fileInput').click()}
          >
            <Upload className="w-10 h-10 text-electric-blue mb-2" />
            <p className="text-lg font-medium mb-2">Drag & drop or click to upload</p>
            <input
              id="fileInput"
              type="file"
              accept="image/*"
              className="hidden"
              onChange={handleFileChange}
            />
          </div>

          {/* Comparison Slider or Loader */}
          <div className="w-full max-w-md mt-4">
            {loading ? (
              <div className="relative h-64 flex items-center justify-center bg-slate-900 rounded-xl">
                <Activity className="animate-spin text-electric-blue w-12 h-12" />
                <span className="absolute bottom-4 left-0 right-0 text-center text-electric-blue font-semibold">
                  Scanning...
                </span>
              </div>
            ) : uploadedImage && heatmapBase64 ? (
              <ReactCompareSlider
                itemOne={
                  <ReactCompareSliderImage
                    src={uploadedImage}
                    alt="Original"
                    style={{ objectFit: 'contain', height: '16rem' }}
                  />
                }
                itemTwo={
                  <ReactCompareSliderImage
                    src={`data:image/png;base64,${heatmapBase64}`}
                    alt="Heatmap"
                    style={{ objectFit: 'contain', height: '16rem' }}
                  />
                }
                className="rounded-xl"
              />
            ) : (
              <div className="h-64 flex items-center justify-center bg-slate-900 rounded-xl text-slate-500">
                Upload an image to see the comparison
              </div>
            )}
          </div>
        </section>

        {/* Right: Results Sidebar */}
        <aside className="w-full md:w-80 bg-slate-900 border-l border-slate-800 p-8 flex flex-col items-center justify-center">
          <div className="mb-8">
            <div className="relative w-32 h-32 flex items-center justify-center">
              <svg className="absolute w-full h-full" viewBox="0 0 100 100">
                <circle
                  cx="50"
                  cy="50"
                  r="45"
                  stroke="#00eaff"
                  strokeWidth="8"
                  fill="none"
                  opacity="0.2"
                />
                <circle
                  cx="50"
                  cy="50"
                  r="45"
                  stroke="#00eaff"
                  strokeWidth="8"
                  fill="none"
                  strokeDasharray={2 * Math.PI * 45}
                  strokeDashoffset={
                    attentionScore !== null
                      ? 2 * Math.PI * 45 * (1 - attentionScore)
                      : 2 * Math.PI * 45
                  }
                  strokeLinecap="round"
                  style={{ transition: 'stroke-dashoffset 1s', transform: 'rotate(-90deg)', transformOrigin: 'center' }}
                />
              </svg>
              <span className="text-3xl font-bold text-electric-blue z-10">
                {attentionScore !== null ? `${Math.round(attentionScore * 100)}%` : '--'}
              </span>
            </div>
            <div className="mt-4 text-center">
              <span className="text-lg font-semibold">Attention Score</span>
            </div>
          </div>
          <div className="w-full">
            <div className="bg-slate-800 rounded-lg p-4 text-center">
              <span className="block text-sm text-slate-400 mb-2">AI Recommendation</span>
              <span className="text-base font-medium">
                {attentionScore !== null
                  ? getRecommendation(attentionScore)
                  : 'Upload an image to get insights.'}
              </span>
            </div>
          </div>
        </aside>
      </main>
    </div>
  );
}

export default App;
