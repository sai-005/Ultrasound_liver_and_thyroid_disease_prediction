import "./App.css";

export default function App() {
  return (
    <div style={styles.container}>
      <iframe
        src="http://127.0.0.1:8001/"
        title="Ultrasound AI Assistant"
        style={styles.iframe}
      />
    </div>
  );
}

const styles = {
  container: {
    width: "100vw",
    height: "100vh",
    backgroundColor: "#020617",
  },
  iframe: {
    width: "100%",
    height: "100%",
    border: "none",
  },
};
