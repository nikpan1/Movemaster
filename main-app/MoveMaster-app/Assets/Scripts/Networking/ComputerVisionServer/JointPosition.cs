
namespace CVServer.Models
{
    public class JointPosition
    {
        public float X { get; set; }
        public float Y { get; set; }
        public bool Flag { get; set; }

        public JointPosition(float x, float y, bool flag)
        {
            this.X = x;
            this.Y = y;
            this.Flag = flag;
        }
    }

}
