# import libraries
import cv2
import time
import numpy as np
import sqlite3
import matplotlib.pyplot as plt

con = sqlite3.connect("traffic.db")
c = con.cursor()
data = c.execute("""SELECT * FROM data""")
rows = c.fetchall()
df = []
df1 = []
df2 = []
df3 = []
for rows in rows:
    df.append(rows[0])
    df1.append(rows[1])
    df2.append(rows[2])
    df3.append(rows[3])
con.commit()
c.close()
con.close()


def capturing(p):
    net = cv2.dnn.readNet("yolo/yolov3.weights", "yolo/yolov3.cfg")
    counter = 1
    with open("yolo/coco.names", "r") as f:
        classes = [line.strip() for line in f.readlines()]
    layer_names = net.getLayerNames()
    output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]
    colors = np.random.uniform(0, 255, size=(len(classes), 3))  # to get list of colors for each possible class
    # Loading image
    with open("{}.jpeg".format(counter), "wb") as f:
        f.write(p)
    frame = cv2.imread("{}.jpeg".format(counter))
    frame = cv2.resize(frame, None, fx=0.4, fy=0.4)
    height, width, channels = frame.shape
    startingtime = time.time()
    frame_id = 0

    # Detecting objects
    blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)

    net.setInput(blob)
    outs = net.forward(output_layers)
    l = ['person', 'car', 'truck', 'bus', 'bike']
    m = dict({'person': 1, 'car': 15, 'truck': 20, 'bus': 20, 'bike': 5})
    # Showing information on the screen
    class_ids = []
    confidences = []
    boxes = []  # coordinate of bounding box
    for out in outs:
        for detection in out:
            scores = detection[5:]  # getting all 80 scores
            class_id = np.argmax(scores)  # finding the max score
            confidence = scores[class_id]
            # find out strong predictions greater then. 5
            if confidence > 0.5:
                # Object detected
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)

                # Rectangle coordinates
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)

                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
    count_label = []
    count = []
    font = cv2.FONT_HERSHEY_PLAIN
    for i in range(len(boxes)):
        if i in indexes:
            x, y, w, h = boxes[i]
            label = str(classes[class_ids[i]])
            if label not in count_label:
                if label in l:
                    count_label.append(label)
                    count.append(int(1))
            else:
                tmp = 0
                for k in count_label:
                    if k == label:
                        count[tmp] = count[tmp] + 1
                    tmp = tmp + 1
            color = colors[class_ids[i]]
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            cv2.putText(frame, label, (x, y + 30), font, 3, color, 3)
    x = 0
    for k in range(len(count_label)):
        x = x + m[count_label[k]]
    elapsed_time = time.time() - startingtime
    fps = frame_id / elapsed_time
    cv2.putText(frame, "FPS:" + str(fps), (10, 30), font, 3, (0, 0, 0), 1)
    cv2.imshow("Image", frame)
    key = cv2.waitKey(1)  # 0 keeps on hold 1 waits for a millisecond
    return x


# define the shape of the environment (i.e., its states)before importing map let's do it in 11 by 11 area
environment_rows = 6
environment_columns = 6

# Create a 3D numpy array to hold the current Q-values for each state and action pair: Q(s, a)
# The array contains 11 rows and 11 columns (to match the shape of the environment),
# as well as a third "action" dimension.
# The "action" dimension consists of 4 layers that will allow us to keep track of
# the Q-values for each possible action in
# each state (see next cell for a description of possible actions).
# The value of each (state, action) pair is initialized to 0.
q_values = np.zeros((environment_rows, environment_columns, 4))
# define actions
# numeric action codes: 0 = up, 1 = right, 2 = down, 3 = left
actions = ['up', 'right', 'down', 'left']
# Create a 2D numpy array to hold the rewards for each state.
# The array contains 11 rows and 11 columns (to match the shape of the environment),
# and each value is initialized to -999999.
rewards = np.full((environment_rows, environment_columns), -999999.)

k = 0
print("Pick the destination Location from the list")
print("Locations :")
for i in range(len(df)):
    k = int(capturing(df[i]))
    if k == 0:
        k = 1
    rewards[df1[i] - 23, df2[i] - 23] = k * (-1)
    print(df3[i])

# taking the value of destination
goalone = -1
goaltwo = -1
goallo = input("Enter Destination Location : ")
for i in range(len(df)):
    if df3[i] == goallo:
        goalone = df1[i] - 23
        goaltwo = df2[i] - 23
if goalone == -1 or goaltwo == -1:
    print("Location not found please check for typos and case if you think u entered correct location")
    exit()

# set the reward for reaching goal (i.e., the goal) to 999999
rewards[goalone, goaltwo] = 999999.0


# define a function that determines if the specified location is a terminal state
def is_terminal_state(current_row_index, current_column_index):
    # if the reward for this location is -1, then it is not a terminal state
    # (i.e., it is a path which we can travel)
    if rewards[current_row_index, current_column_index] == 999999.0 or rewards[
        current_row_index, current_column_index] == -999999.0:
        return True
    else:
        return False


# define a function that will choose a random, non-terminal starting location
def get_starting_location():
    # get a random row and column index
    current_row_index = np.random.randint(environment_rows)
    current_column_index = np.random.randint(environment_columns)
    # continue choosing random row and column indexes until a non-terminal state is identified
    # (i.e., until the chosen state is a 'path which we can travel').
    while is_terminal_state(current_row_index, current_column_index):
        current_row_index = np.random.randint(environment_rows)
        current_column_index = np.random.randint(environment_columns)
    return current_row_index, current_column_index


# define an epsilon greedy algorithm that will choose which action to take next (i.e., where to move next)
def get_next_action(current_row_index, current_column_index, epsilon):
    # if a randomly chosen value between 0 and 1 is less than epsilon,
    # then choose the most promising value from the Q-table for this state.
    if np.random.random() < epsilon:
        return np.argmax(q_values[current_row_index, current_column_index])
    else:  # choose a random action
        return np.random.randint(4)


# define a function that will get the next location based on the chosen action
def get_next_location(current_row_index, current_column_index, action_index):
    new_row_index = current_row_index
    new_column_index = current_column_index
    if actions[action_index] == 'up' and current_row_index > 0:
        new_row_index -= 1
    elif actions[action_index] == 'right' and current_column_index < environment_columns - 1:
        new_column_index += 1
    elif actions[action_index] == 'down' and current_row_index < environment_rows - 1:
        new_row_index += 1
    elif actions[action_index] == 'left' and current_column_index > 0:
        new_column_index -= 1
    return new_row_index, new_column_index


# Define a function that will get the shortest path between any location within the source that
# the car is allowed to travel and the goal.
def get_shortest_path(start_row_index, start_column_index):
    # return immediately if this is an invalid starting location
    if is_terminal_state(start_row_index, start_column_index):
        print("You are not on road please get to the road first")
        return []
    else:  # if this is a 'legal' starting location
        current_row_index, current_column_index = start_row_index, start_column_index
        shortest_path = []
        shortest_path.append([current_row_index, current_column_index])
    # continue moving along the path until we reach the goal (i.e., the item packaging location)
    while not is_terminal_state(current_row_index, current_column_index):
        # get the best action to take
        action_index = get_next_action(current_row_index, current_column_index, 1.)
        # move to the next location on the path, and add the new location to the list
        current_row_index, current_column_index = get_next_location(current_row_index, current_column_index,
                                                                    action_index)
        shortest_path.append([current_row_index, current_column_index])
    return shortest_path


# define training parameters
epsilon = 0.9  # the percentage of time when we should take the best action (instead of a random action)
discount_factor = 0.9  # discount factor for future rewards
learning_rate = 0.9  # the rate at which the AI agent should learn

# run through 1000 training episodes
for episode in range(1000):
    # get the starting location for this episode
    row_index, column_index = get_starting_location()

    # continue taking actions (i.e., moving) until we reach a terminal state
    # (i.e., until we reach goal or crash )
    while not is_terminal_state(row_index, column_index):
        # choose which action to take (i.e., where to move next)
        action_index = get_next_action(row_index, column_index, epsilon)

        # perform the chosen action, and transition to the next state (i.e., move to the next location)
        old_row_index, old_column_index = row_index, column_index  # store the old row and column indexes
        row_index, column_index = get_next_location(row_index, column_index, action_index)

        # receive the reward for moving to the new state, and calculate the temporal difference
        reward = rewards[row_index, column_index]
        old_q_value = q_values[old_row_index, old_column_index, action_index]
        temporal_difference = reward + (discount_factor * np.max(q_values[row_index, column_index])) - old_q_value

        # update the Q-value for the previous state and action pair
        new_q_value = old_q_value + (learning_rate * temporal_difference)
        q_values[old_row_index, old_column_index, action_index] = new_q_value

print('Training complete!')

sourceone = -1
sourcetwo = -1
sourcelo = input("Enter the source from same list Location : ")
for i in range(len(df)):
    if df3[i] == sourcelo:
        sourceone = df1[i] - 23
        sourcetwo = df2[i] - 23
if sourceone == -1 or sourcetwo == -1:
    print("Location not found please check for typos and case if you think u entered correct location")
    exit()
q = get_shortest_path(sourceone, sourcetwo)
q1 = []
if q == q1:
    print("Your are on the Destination :")
    exit()
row = np.array(q)
x = []
y = []
for i in range(len(row)):
    x.append(23 + row[i][0])
    y.append(23 + row[i][1])

for i in range(len(x) - 1):
    for j in range(len(df)):
        if df1[j] == x[i] and df2[j] == y[i]:
            print(df3[j], "-->", end=" ")
print(goallo)

x = []
y = []
for i in range(len(row)):
    x.append(row[i][0])
    y.append(row[i][1])

# Plotting the Graph
plt.scatter(x, y)
plt.plot(x, y)
plt.xlabel("Latitude (in Minutes X 10^2)")
plt.ylabel("Longitude (in Minutes X 10^2)")
plt.show()
cv2.destroyAllWindows()