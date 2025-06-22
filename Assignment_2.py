class Node:
    def __init__(self, data):
        self.data = data  
        self.next = None  

class LinkedList:
    def __init__(self):
        self.head = None  # list empty initially

    # Adds new node at end of list
    def append(self, data):
        new_node = Node(data)
        if self.head is None:
            #list empty, new node is head
            self.head = new_node
        else:
          
            current = self.head
            while current.next:
                current = current.next
            current.next = new_node
    #prints the list
    def printList(self):
        if self.head is None:
            print("List is empty")
            return

        current = self.head
        print("Linked List:", end=" ")
        while current:
            print(current.data, end=" -> ")
            current = current.next
        print("None")
    #deletes nth node
    def delete_nthNode(self, n):
        try:
            if self.head is None:
                raise Exception("Can't delete from empty list")

            if n <= 0:
                raise IndexError("Index should be positive integer")

            if n == 1:
                deletedData = self.head.data
                self.head = self.head.next
                print(f"Node deleted: {deletedData}")
                return

            current = self.head
            count = 1
            while current and count < n - 1:
                current = current.next
                count += 1

            if current is None or current.next is None:
                raise IndexError("Index out of range")

            deletedData = current.next.data
            current.next = current.next.next
            print(f"Node deleted: {deletedData}")

        except Exception as e:
            print("Error:", e)
#menu driven
def menu():
    list = LinkedList()
    print("Choose from list:\n 1.Append\n 2.Print List\n 3.Delete nth Node\n 4.Exit")
    
    while True:
        choice = int(input("\nEnter your choice(1-4): "))
        if choice == 1:
            data = input("Enter data to append: ")
            list.append(data)
        elif choice == 2:
            list.printList()
        elif choice == 3:
            n = int(input("Enter position of node to delete: "))
            list.delete_nthNode(n)
        elif choice == 4:
            print("Exiting...")
            break
        else:
            print("Incorrect choice, please enter a number between 1-4")

if __name__ == "__main__":
    menu()