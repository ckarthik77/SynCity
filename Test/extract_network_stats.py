import sumolib

# Load your network
net = sumolib.net.readNet('E:\Desktop\Jobs\projects\Syncity\Test\city_tls.net.xml')

# Extract statistics
print("=== SUMO Network Statistics ===")
print(f"Total Junctions: {len(net.getNodes())}")
print(f"Total Edges: {len(net.getEdges())}")

# Count lanes
total_lanes = sum(len(edge.getLanes()) for edge in net.getEdges())
print(f"Total Lanes: {total_lanes}")

# Count traffic lights
tls_count = len(net.getTrafficLights())
print(f"Traffic Lights: {tls_count}")

# Calculate network area
nodes = net.getNodes()
lats = [node.getCoord()[1] for node in nodes]
lons = [node.getCoord()[0] for node in nodes]
print(f"Bounding Box: ({min(lons)}, {min(lats)}) to ({max(lons)}, {max(lats)})")

# Average edge length
edge_lengths = [edge.getLength() for edge in net.getEdges()]
avg_length = sum(edge_lengths) / len(edge_lengths)
print(f"Average Edge Length: {avg_length:.2f} m")