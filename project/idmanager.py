import math
from random import randint
import cv2

class IDManager:
    
    class ID:
    
        def __init__(self, position, IDnum):
            """ An ID is definied by its x, y position"""
            self.IDnum = IDnum
            (self.x, self.y) = position
        
        def update_position(self, position):
            (self.x, self.y) = position
        
        def get_position(self):
            return (self.x,self.y)

    def __init__(self):
        """ 
            The manager will keep track of all the active ID's
            by using a list of active IDs. 
        """
        self.IDS = []
        self.IDnum = 0

    def createID(self, position):
        """
            When given a new position, increment the IDnum
            and assign the new ID a position. 
        """
        self.IDnum += 1
        self.IDS.append(self.ID(position, self.IDnum))
        
    def updatePositions(self, possiblePositions):
        """
        Compute the distances from each ID to all the possible 
        ID points. The lowest distance one is probably the same ID,
        so update the ID's position to the smallest delta_d. 
        """
        # For every ID we are tracking
        for id in self.IDS:
            distances = []
            # Look at every possible point the ID could have moved to 
            for possiblePosition in possiblePositions:
                # Compute the distance from every point and add to a list
                (x_final, y_final) = possiblePosition
                distances.append(math.sqrt((x_final - id.x)**2 + (y_final - id.y)**2))
            
            # Take the lowest distance computed and use that as the new ID position
            if distances != []:
                lowest_index = distances.index(min(distances)) 
                updated_position = possiblePositions[lowest_index]
            
                # Remove the position this ID claims, to lessen computations done in future rounds
                del possiblePositions[lowest_index]

                id.update_position(updated_position)
    def draw_ids(self, img):
        for id in self.IDS:
            img = cv2.putText(img, str(id.IDnum), (int(id.x), int(id.y)), fontFace=cv2.FONT_HERSHEY_TRIPLEX, fontScale=1, color=(255, 255, 255),thickness=1)
        return img
            


if __name__ == '__main__':

    IDS = IDManager()

    num = input('How many drones?')
    print()

    initial_positions = []
    for x in range(int(num)):
        print(f"Drone Num: {x}")
        x = input('Initial X?')
        y = input('Initial Y?')
        print()

        IDS.createID((int(x),int(y)))
    
    num = input("How many possible points do you want to give?")

    possiblePositions = []
    for x in range(int(num)):
        possiblePositions.append((randint(0,20), randint(0,20)))

    print()
    print("Starting Positions")
    for id in IDS.IDS:
        print(id.get_position())

    print()
    print("Possible Possitions")
    print(possiblePositions)

    IDS.updatePositions(possiblePositions)

    print()
    print("Updated Possitions")
    for id in IDS.IDS:
        print(f"ID {id.IDnum} is at {id.get_position()}")
    


    



        

                


                







"""
    Assign an ID to the drone at the beginning. 
    The ID begins with the postiion of the drone.

    To Update the ID, pass a list of potential points
    and select the point that is closest to the ID.
    Update the ID's position.  
"""

"""
    Rather than passing the points to each ID, 
    Make a manager that will figure it out..

    Collect a list of the drones positions, 

    To Update, pass the lists of potential points, 

"""