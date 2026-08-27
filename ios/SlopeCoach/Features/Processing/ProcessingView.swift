import SwiftUI

struct ProcessingView: View {
    var body: some View {
        PlaceholderFeatureView(
            title: "Analyzing...",
            systemImage: "waveform.path.ecg"
        )
        .navigationTitle("Processing")
        .navigationBarTitleDisplayMode(.inline)
    }
}
