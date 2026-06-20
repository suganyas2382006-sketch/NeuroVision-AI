<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>NeuroVision AI | Brain Tumor Analysis</title>
    <!-- Tailwind CSS CDN -->
    <script src="https://cdn.jsdelivr.net/npm/@tailwindcss/browser@4"></script>
    <!-- FontAwesome Icons -->
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css">
</head>
<body class="bg-slate-900 text-slate-100 min-h-screen font-sans antialiased">

    <!-- Top Navigation Bar -->
    <nav class="border-b border-slate-800 bg-slate-900/80 backdrop-blur sticky top-0 z-50 px-6 py-3 flex items-center justify-between">
        <div class="flex items-center gap-3">
            <!-- App Logo Custom Integration -->
            <div class="w-10 h-10 rounded-lg overflow-hidden border border-slate-700 bg-slate-800 flex items-center justify-center">
                <img src="/static/images/1000195327.jpg" alt="NeuroVision AI Logo" class="w-full h-full object-cover">
            </div>
            <div>
                <h1 class="text-lg font-bold tracking-tight">NeuroVision <span class="text-indigo-400">AI</span></h1>
                <p class="text-xs text-slate-400">MRI Diagnostics & Severity Suite</p>
            </div>
        </div>
        <div class="flex items-center gap-4 text-sm text-slate-400">
            <span class="flex items-center gap-1"><i class="fa-solid fa-circle text-emerald-500 text-[10px]"></i> AI Engine Active</span>
        </div>
    </nav>

    <!-- Main Workspace -->
    <main class="max-w-7xl mx-auto p-6 grid grid-cols-1 lg:grid-cols-3 gap-6">
        
        <!-- Left Column: Controls & Upload -->
        <div class="space-y-6">
            <div class="bg-slate-800/50 border border-slate-700/60 rounded-xl p-5 backdrop-blur-sm">
                <h2 class="text-md font-semibold mb-3 flex items-center gap-2"><i class="fa-solid fa-upload text-indigo-400"></i> Patient Data Input</h2>
                <p class="text-xs text-slate-400 mb-4">Upload a high-resolution T1, T2, or FLAIR weighted brain MRI scan.</p>
                
                <form id="uploadForm" enctype="multipart/form-data" class="space-y-4">
                    <label class="border-2 border-dashed border-slate-600 hover:border-indigo-500 transition-colors rounded-xl p-8 flex flex-col items-center justify-center cursor-pointer group bg-slate-900/40">
                        <i class="fa-solid fa-cloud-arrow-up text-3xl text-slate-500 group-hover:text-indigo-400 transition-colors mb-2"></i>
                        <span class="text-sm font-medium text-slate-300">Choose MRI Scan File</span>
                        <span class="text-xs text-slate-500 mt-1">Supports JPEG, PNG</span>
                        <input type="file" name="mri_image" id="fileInput" class="hidden" accept="image/*" required>
                    </label>

                    <!-- Preview of Selected Image File -->
                    <div id="fileSelectedName" class="hidden text-xs bg-slate-900 border border-slate-700 p-2 rounded text-slate-300 flex items-center justify-between">
                        <span class="truncate pr-4"><i class="fa-solid fa-file-image text-indigo-400 mr-1"></i> <span id="fileNameSpan"></span></span>
                        <button type="button" id="clearFile" class="text-rose-400 hover:text-rose-300"><i class="fa-solid fa-xmark"></i></button>
                    </div>

                    <button type="submit" class="w-full bg-indigo-600 hover:bg-indigo-500 text-white font-medium py-2.5 px-4 rounded-lg transition-colors flex items-center justify-center gap-2 shadow-lg shadow-indigo-600/20 cursor-pointer">
                        <i class="fa-solid fa-wand-magic-sparkles"></i> Run AI Diagnosis
                    </button>
                </form>
            </div>
        </div>

        <!-- Middle & Right Columns: Visualizations & Analytics -->
        <div class="lg:col-span-2 space-y-6">
            
            <!-- Default Welcome State -->
            <div id="emptyState" class="bg-slate-800/30 border border-slate-700/40 border-dashed rounded-xl p-16 flex flex-col items-center justify-center text-center h-[500px]">
                <div class="w-16 h-16 bg-slate-800 rounded-full flex items-center justify-center text-slate-500 mb-4">
                    <i class="fa-solid fa-stethoscope text-2xl"></i>
                </div>
                <h3 class="text-lg font-medium text-slate-300">Awaiting Patient Data</h3>
                <p class="text-sm text-slate-500 max-w-sm mt-1">Please select and execute an analysis on an MRI scan file from the command console on the left.</p>
            </div>

            <!-- Analysis Dashboard (Initially Hidden) -->
            <div id="dashboardState" class="hidden space-y-6">
                
                <!-- Metrics Bar -->
                <div class="grid grid-cols-2 sm:grid-cols-4 gap-4">
                    <div class="bg-slate-800/60 border border-slate-700/50 p-4 rounded-xl">
                        <span class="text-xs font-medium text-slate-400 block mb-1">Detection Result</span>
                        <span id="metricPrediction" class="text-sm font-bold text-rose-400">—</span>
                    </div>
                    <div class="bg-slate-800/60 border border-slate-700/50 p-4 rounded-xl">
                        <span class="text-xs font-medium text-slate-400 block mb-1">AI Confidence</span>
                        <span id="metricConfidence" class="text-sm font-bold text-emerald-400">—</span>
                    </div>
                    <div class="bg-slate-800/60 border border-slate-700/50 p-4 rounded-xl">
                        <span class="text-xs font-medium text-slate-400 block mb-1">Severity Grading</span>
                        <span id="metricSeverity" class="text-sm font-bold text-amber-400">—</span>
                    </div>
                    <div class="bg-slate-800/60 border border-slate-700/50 p-4 rounded-xl">
                        <span class="text-xs font-medium text-slate-400 block mb-1">Inference Latency</span>
                        <span id="metricTime" class="text-sm font-bold text-slate-300">—</span>
                    </div>
                </div>

                <!-- Imaging Viewports -->
                <div class="grid grid-cols-1 md:grid-cols-2 gap-6">
                    <!-- Source Frame -->
                    <div class="bg-slate-800/50 border border-slate-700/60 rounded-xl p-4">
                        <h3 class="text-xs font-semibold text-slate-400 mb-3 uppercase tracking-wider"><i class="fa-solid fa-microscope text-indigo-400 mr-1"></i> Original MRI Input</h3>
                        <div class="bg-slate-900 rounded-lg aspect-square overflow-hidden border border-slate-800 flex items-center justify-center">
                            <img id="viewOriginal" src="" alt="Original MRI" class="w-full h-full object-contain">
                        </div>
                    </div>
                    <!-- XAI Heatmap Frame -->
                    <div class="bg-slate-800/50 border border-slate-700/60 rounded-xl p-4">
                        <h3 class="text-xs font-semibold text-slate-400 mb-3 uppercase tracking-wider"><i class="fa-solid fa-bolt text-amber-400 mr-1"></i> Explainable AI (Grad-CAM)</h3>
                        <div class="bg-slate-900 rounded-lg aspect-square overflow-hidden border border-slate-800 flex items-center justify-center relative">
                            <img id="viewHeatmap" src="" alt="XAI Heatmap" class="w-full h-full object-contain mix-blend-screen opacity-90">
                            <div class="absolute bottom-2 right-2 bg-slate-950/80 backdrop-blur px-2 py-1 rounded text-[10px] text-slate-400 border border-slate-800">
                                High Attention Map
                            </div>
                        </div>
                    </div>
                </div>

                <!-- Functional Reporting Actions -->
                <div class="flex justify-end gap-3 pt-2">
                    <button onclick="window.location.reload()" class="bg-slate-800 hover:bg-slate-700 border border-slate-700 text-sm font-medium py-2 px-4 rounded-lg transition-colors flex items-center gap-2 cursor-pointer">
                        <i class="fa-solid fa-rotate-left"></i> Reset Dashboard
                    </button>
                    <button class="bg-emerald-600 hover:bg-emerald-500 text-white text-sm font-medium py-2 px-4 rounded-lg transition-colors flex items-center gap-2 shadow-lg shadow-emerald-600/10 cursor-pointer">
                        <i class="fa-solid fa-file-pdf"></i> Generate Export PDF Report
                    </button>
                </div>

            </div>

        </div>
    </main>

    <script>
        const fileInput = document.getElementById('fileInput');
        const fileSelectedName = document.getElementById('fileSelectedName');
        const fileNameSpan = document.getElementById('fileNameSpan');
        const clearFile = document.getElementById('clearFile');
        const uploadForm = document.getElementById('uploadForm');

        const emptyState = document.getElementById('emptyState');
        const dashboardState = document.getElementById('dashboardState');

        fileInput.addEventListener('change', (e) => {
            if(e.target.files.length > 0) {
                fileNameSpan.textContent = e.target.files[0].name;
                fileSelectedName.classList.remove('hidden');
            }
        });

        clearFile.addEventListener('click', () => {
            fileInput.value = '';
            fileSelectedName.classList.add('hidden');
        });

        uploadForm.addEventListener('submit', async (e) => {
            e.preventDefault();
            const formData = new FormData(uploadForm);
            
            try {
                const response = await fetch('/analyze', {
                    method: 'POST',
                    body: formData
                });
                const data = await response.json();
                
                if (data.success) {
                    document.getElementById('metricPrediction').textContent = data.metrics.prediction;
                    document.getElementById('metricConfidence').textContent = data.metrics.confidence;
                    document.getElementById('metricSeverity').textContent = data.metrics.severity;
                    document.getElementById('metricTime').textContent = data.metrics.analysis_time;
                    
                    document.getElementById('viewOriginal').src = data.image_url;
                    document.getElementById('viewHeatmap').src = data.heatmap_url;

                    emptyState.classList.add('hidden');
                    dashboardState.classList.remove('hidden');
                } else {
                    alert("Analysis error: " + data.error);
                }
            } catch (err) {
                console.error(err);
                alert("An error occurred executing the inference model pipeline.");
            }
        });
    </script>
</body>
</html>
