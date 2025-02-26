# 🚀 Python Path Planning with Ant Colony Optimization

This project implements a **path planning algorithm** using **Ant Colony Optimization (ACO)** for the **Traveling Salesman Problem (TSP)**. It also includes **lawnmower motion strategies** to efficiently cover the mapped area. The algorithm generates **random regular polygons**, divides the area, and optimizes the traversal path.


THis Project and Documentation is done by Mr. Gaurav Pratap Singh,Mr. Lokesh and Mr. Jating Garg under the Supervision of assistant processor Mr. Kunwar Pal in NIT Jalandhar as a Major Project.
---

## 📌 Features  
✅ **Random Polygon Generation** – Generates random regular polygons within a defined area.  
✅ **Lawnmower Motion Strategies** – Implements different traversal patterns for area coverage.  
✅ **Ant Colony Optimization (ACO)** – Solves TSP for finding the shortest path.  
✅ **Matplotlib 3D Visualization** – Displays polygons, paths, and movements in a 3D space.  
✅ **Shapely Geometry Processing** – Ensures accurate polygonal computations.  

---

## 🛠️ Installation & Setup  

### 🔹 Prerequisites  
- Python 3.x  
- Required libraries (install using `requirements.txt`)  

### 🔹 Clone the Repository  
```bash
git clone https://github.com/Mrgaurav07/Drone-Path-Planning.git
# install the dependencies
pip install -r requirements.txt

# run the main program
python 3Dimplementation.py


📊 Algorithm Details
1️⃣ Random Polygon Generation
Generates random regular polygons inside a predefined area.
Uses Shapely to handle polygon operations.
2️⃣ Lawnmower Motion Strategies
Implements four different traversal patterns for full area coverage.
Ensures optimized movement inside the given space.
3️⃣ Ant Colony Optimization (ACO) for TSP
Uses Ant Colony Optimization (ACO) to find the shortest route.
Optimizes the order in which centroids of polygons are visited.
Utilizes pheromone-based decision making to improve path efficiency.
📊 Performance Metrics
🚀 Optimized for efficiency, ensuring:
✅ Fast computation of paths.
✅ Reduced energy consumption.
✅ Collision-free navigation.

📜 License
This project is licensed under the MIT License – feel free to use and modify.

🙌 Contributing
Contributions are welcome! Feel free to fork, create issues, or submit pull requests.

📩 Contact: [Your Email] | 🔗 GitHub: [Your GitHub Profile]

🌟 Star this repo if you find it useful! ⭐
yaml
Copy
Edit

---

### **📌 requirements.txt**  

```txt
matplotlib
shapely
numpy

