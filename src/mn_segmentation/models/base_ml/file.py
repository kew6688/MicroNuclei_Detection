class CellImage:
    def __init__(self, name):
        self.name = name
        self.expi_cond = ""

    def extract(self):
        self.name.split("_")

    def delNodes(self, root: Optional[TreeNode], to_delete: List[int]) -> List[TreeNode]:
        
            
        st = []
        for i,v in enumerate(nums):
            while v >= st[-1][1]:
                ind,_ = st.pop()
                dp[i] = min(dp[i], cost[i] + dp[ind])
            st.append((i,v))

        st = []
        for i,v in enumerate(nums):
            while v < st[-1][1]:
                ind,_ = st.pop()
                dp[i] = min(dp[i], cost[i] + dp[ind])
            st.append((i,v))
        return dp[-1]
