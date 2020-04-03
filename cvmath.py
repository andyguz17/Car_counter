#Math

class dot: 

    def __init__(self,x,y):
        self.x = x
        self.y = y

    def __str__(self):
        return "("+str(self.x)+","+str(self.y)+")"

class line: 

    def __init__(self,p0,p1):
        self.p0 = p0
        self.p1 = p1
        self.A = p1.x - p0.x
        self.B = p1.y - p0.y
        self.C = p1.x*p0.y - p0.x*p1.y
    
    def fx (self,x):

        straight = ((self.B)*x+self.C)/self.A
        return  straight

    def intersect(self, other):

        det = self.A*other.B - other.A*self.B
        x = other.A*self.C-self.A*other.C
        y = other.B*self.C-self.B*other.C
        if (det != 0):
        
            return True, dot(x/det,y/det)
    
        else: 
            
            return False,-1 
