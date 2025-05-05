
## Detectron2 Demo

We provide a command line tool to run a simple demo of builtin configs.
The usage is explained in [GETTING_STARTED.md](../GETTING_STARTED.md).

See our [blog post](https://ai.facebook.com/blog/-detectron2-a-pytorch-based-modular-object-detection-library-)
for a high-quality demo ge<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 800 580">
  <!-- Background -->
  <rect width="800" height="580" fill="white"/>
  
  <!-- Title moved higher -->
  <text x="400" y="15" font-family="Arial" font-size="20" font-weight="bold" text-anchor="middle">Visual SLAM System with Adaptive Algorithm Switching</text>
  <line x1="100" y1="18" x2="700" y2="18" stroke="#dddddd" stroke-width="1"/>
  
  <!-- Input Source -->
  <rect x="50" y="75" width="140" height="80" rx="5" ry="5" fill="#e6f2ff" stroke="#3366cc" stroke-width="2"/>
  <text x="120" y="105" font-family="Arial" font-size="14" font-weight="bold" text-anchor="middle">Intel RealSense</text>
  <text x="120" y="125" font-family="Arial" font-size="12" text-anchor="middle">Depth + RGB Images</text>
  
  <!-- Input description -->
  <rect x="20" y="165" width="200" height="40" rx="5" ry="5" fill="#f5f5f5" stroke="#3366cc" stroke-width="1" stroke-dasharray="3,2"/>
  <text x="120" y="190" font-family="Arial" font-size="10" font-style="italic" text-anchor="middle">High-resolution depth and RGB data</text>
  
  <!-- RE-RANSAC-ICP Path -->
  <rect x="250" y="50" width="150" height="60" rx="5" ry="5" fill="#e6ffe6" stroke="#339933" stroke-width="2"/>
  <text x="325" y="75" font-family="Arial" font-size="14" font-weight="bold" text-anchor="middle">RE-RANSAC-ICP</text>
  <text x="325" y="95" font-family="Arial" font-size="12" text-anchor="middle">(Depth Image)</text>
  
  <!-- RE-RANSAC description -->
  <rect x="250" y="20" width="150" height="25" rx="5" ry="5" fill="#f5f5f5" stroke="#339933" stroke-width="1" stroke-dasharray="3,2"/>
  <text x="325" y="35" font-family="Arial" font-size="10" font-style="italic" text-anchor="middle">Robust in low-texture areas</text>
  
  <!-- 8Point RANSAC Path -->
  <rect x="250" y="130" width="150" height="60" rx="5" ry="5" fill="#e6ffe6" stroke="#339933" stroke-width="2"/>
  <text x="325" y="155" font-family="Arial" font-size="14" font-weight="bold" text-anchor="middle">8Point RANSAC</text>
  <text x="325" y="175" font-family="Arial" font-size="12" text-anchor="middle">(RGB Image)</text>
  
  <!-- 8Point description -->
  <rect x="250" y="195" width="150" height="25" rx="5" ry="5" fill="#f5f5f5" stroke="#339933" stroke-width="1" stroke-dasharray="3,2"/>
  <text x="325" y="210" font-family="Arial" font-size="10" font-style="italic" text-anchor="middle">Better in feature-rich scenes</text>
  
  <!-- Arrows from Input to Algorithms -->
  <line x1="190" y1="100" x2="250" y2="80" stroke="black" stroke-width="2"/>
  <polygon points="250,80 240,77 242,83" fill="black"/>
  
  <line x1="190" y1="120" x2="250" y2="150" stroke="black" stroke-width="2"/>
  <polygon points="250,150 240,147 242,153" fill="black"/>
  
  <!-- Switching Component -->
  <path d="M 450,110 L 500,70 L 550,110 L 500,150 Z" fill="#fff2e6" stroke="#ff9933" stroke-width="2"/>
  <text x="500" y="115" font-family="Arial" font-size="14" font-weight="bold" text-anchor="middle">Switching</text>
  
  <!-- Switching description -->
  <rect x="425" y="155" width="150" height="65" rx="5" ry="5" fill="#fff9f2" stroke="#ff9933" stroke-width="1"/>
  <text x="500" y="170" font-family="Arial" font-size="10" font-weight="bold" text-anchor="middle">Selection Methods:</text>
  <text x="500" y="185" font-family="Arial" font-size="9" text-anchor="middle">• Quality metrics evaluation</text>
  <text x="500" y="200" font-family="Arial" font-size="9" text-anchor="middle">• Environment detection</text>
  <text x="500" y="215" font-family="Arial" font-size="9" text-anchor="middle">• Confidence weighting</text>
  
  <!-- Arrows from Algorithms to Switch -->
  <line x1="400" y1="80" x2="450" y2="100" stroke="black" stroke-width="2"/>
  <polygon points="450,100 440,95 445,105" fill="black"/>
  
  <line x1="400" y1="150" x2="450" y2="120" stroke="black" stroke-width="2"/>
  <polygon points="450,120 440,125 445,115" fill="black"/>
  
  <!-- Map Components -->
  <rect x="600" y="50" width="150" height="60" rx="5" ry="5" fill="#e6ffe6" stroke="#339933" stroke-width="2"/>
  <text x="675" y="75" font-family="Arial" font-size="14" font-weight="bold" text-anchor="middle">RGB-D BA MAP</text>
  <text x="675" y="95" font-family="Arial" font-size="12" text-anchor="middle">(With Depth)</text>
  
  <!-- RGB-D BA description -->
  <rect x="600" y="20" width="150" height="25" rx="5" ry="5" fill="#f5f5f5" stroke="#339933" stroke-width="1" stroke-dasharray="3,2"/>
  <text x="675" y="35" font-family="Arial" font-size="10" font-style="italic" text-anchor="middle">Accurate depth-aware mapping</text>
  
  <rect x="600" y="130" width="150" height="60" rx="5" ry="5" fill="#e6ffe6" stroke="#339933" stroke-width="2"/>
  <text x="675" y="155" font-family="Arial" font-size="14" font-weight="bold" text-anchor="middle">RGB-BA MAP</text>
  <text x="675" y="175" font-family="Arial" font-size="12" text-anchor="middle">(With RGB)</text>
  
  <!-- RGB BA description -->
  <rect x="600" y="195" width="150" height="25" rx="5" ry="5" fill="#f5f5f5" stroke="#339933" stroke-width="1" stroke-dasharray="3,2"/>
  <text x="675" y="210" font-family="Arial" font-size="10" font-style="italic" text-anchor="middle">Feature-based reconstruction</text>
  
  <!-- Arrows from Switch to Maps -->
  <line x1="550" y1="90" x2="600" y2="80" stroke="black" stroke-width="2"/>
  <polygon points="600,80 590,77 592,83" fill="black"/>
  
  <line x1="550" y1="130" x2="600" y2="150" stroke="black" stroke-width="2"/>
  <polygon points="600,150 590,147 592,153" fill="black"/>
  
  <!-- Global Map -->
  <rect x="400" y="260" width="200" height="60" rx="5" ry="5" fill="#ffe6e6" stroke="#cc3333" stroke-width="2"/>
  <text x="500" y="285" font-family="Arial" font-size="14" font-weight="bold" text-anchor="middle">Global Map</text>
  <text x="500" y="305" font-family="Arial" font-size="12" text-anchor="middle">With Loop Closure</text>
  
  <!-- Global Map description -->
  <rect x="400" y="325" width="200" height="40" rx="5" ry="5" fill="#f5f5f5" stroke="#cc3333" stroke-width="1" stroke-dasharray="3,2"/>
  <text x="500" y="345" font-family="Arial" font-size="10" font-style="italic" text-anchor="middle">Detects revisited areas and refines map</text>
  <text x="500" y="360" font-family="Arial" font-size="10" font-style="italic" text-anchor="middle">for global consistency</text>
  
  <!-- Arrows to Global Map -->
  <line x1="650" y1="220" x2="540" y2="260" stroke="black" stroke-width="2"/>
  <polygon points="540,260 548,255 550,265" fill="black"/>
  
  <line x1="675" y1="220" x2="675" y2="240" stroke="black" stroke-width="2" stroke-dasharray="5,3"/>
  <line x1="675" y1="240" x2="600" y2="280" stroke="black" stroke-width="2" stroke-dasharray="5,3"/>
  <polygon points="600,280 608,275 610,285" fill="black"/>
  <text x="690" y="240" font-family="Arial" font-size="10" font-style="italic">Optional path</text>
  
  <!-- Bundle Adjustment Info -->
  <rect x="610" y="260" width="130" height="50" rx="5" ry="5" fill="#f5f5f5" stroke="#339933" stroke-width="1" stroke-dasharray="3,2"/>
  <text x="675" y="275" font-family="Arial" font-size="10" font-weight="bold" text-anchor="middle">Bundle Adjustment (BA)</text>
  <text x="675" y="290" font-family="Arial" font-size="9" text-anchor="middle">Optimizes camera poses</text>
  <text x="675" y="305" font-family="Arial" font-size="9" text-anchor="middle">and 3D point positions</text>
  
  <!-- Parameter Settings Box -->
  <rect x="250" y="380" width="300" height="130" rx="5" ry="5" fill="white" stroke="#666666" stroke-width="1.5"/>
  <text x="400" y="400" font-family="Arial" font-size="14" font-weight="bold" text-anchor="middle">Intel RealSense Parameters</text>
  
  <text x="270" y="425" font-family="Arial" font-size="12" text-anchor="start">• Horizontal FOV: 1.047</text>
  <text x="270" y="445" font-family="Arial" font-size="12" text-anchor="start">• Image Resolution: 320×240</text>
  <text x="270" y="465" font-family="Arial" font-size="12" text-anchor="start">• Modified in intel-xacro file</text>
  <text x="270" y="485" font-family="Arial" font-size="12" text-anchor="start">• Optimized for real-time SLAM performance</text>
  
  <!-- Arrow from Parameters to System -->
  <line x1="400" y1="380" x2="400" y2="330" stroke="#666666" stroke-width="1.5" stroke-dasharray="5,3"/>
  <polygon points="400,330 395,340 405,340" fill="#666666"/>
  
  <!-- Overall System Description -->
  <rect x="50" y="520" width="700" height="50" rx="5" ry="5" fill="#f8f8f8" stroke="#666666" stroke-width="1"/>
  <text x="400" y="540" font-family="Arial" font-size="11" text-anchor="middle">This SLAM system adaptively selects between two parallel algorithms based on environmental conditions,</text>
  <text x="400" y="555" font-family="Arial" font-size="11" text-anchor="middle">creating optimal maps that combine RGB and depth data to produce a consistent global representation.</text>
</svg>
![enhanced-slam-flowchart-2](https://github.com/user-attachments/assets/5c8c415c-1147-4dfa-9f13-bf0b059afe9b)
nerated with this tool.

![Screenshot from 2025-05-05 00-49-06](https://github.com/user-attachments/assets/eab5c560-a4ed-4bbc-b411-7391d50e3494)
