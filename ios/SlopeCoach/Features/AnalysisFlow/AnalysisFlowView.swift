import SwiftUI

struct AnalysisFlowView: View {
    @State private var viewModel = RiderSelectionViewModel()
    @State private var showsProcessing = false

    var body: some View {
        VStack(spacing: 0) {
            VStack(spacing: 10) {
                RiderVideoPreview(
                    imageName: previewImageName,
                    sourceSize: viewModel.sourcePixelSize,
                    selectionState: viewModel.selectionState,
                    currentMask: viewModel.currentMask,
                    tapPoint: viewModel.tapPoint,
                    pulseID: viewModel.pulseID,
                    isScrubbing: viewModel.isScrubbing
                ) { point in
                    viewModel.selectRider(at: point)
                }

                playbackControls

                VideoTimeline(
                    imageName: previewImageName,
                    currentTime: viewModel.currentTime,
                    duration: viewModel.duration,
                    onScrubBegan: viewModel.beginScrubbing,
                    onScrubChanged: viewModel.scrub(to:),
                    onScrubEnded: viewModel.endScrubbing
                )

                RiderSelectionStatus(
                    state: viewModel.selectionState,
                    selectedTimestamp: viewModel.formattedSelectionTime,
                    onChange: viewModel.changeSelection,
                    onTryAgain: viewModel.tryAgain,
                    onReselect: viewModel.reselectAfterTrackingLoss
                )
            }
            .padding(.horizontal, 16)
            .padding(.top, 8)

            Spacer(minLength: 12)
        }
        .background(Color(.systemBackground).ignoresSafeArea())
        .safeAreaInset(edge: .bottom, spacing: 0) {
            continueButton
        }
        .navigationBarTitleDisplayMode(.inline)
        .toolbar {
            ToolbarItem(placement: .principal) {
                VStack(spacing: 1) {
                    Text("Select Rider")
                        .font(.headline)
                    Text("Tap the skier you want to analyze")
                        .font(.caption)
                        .foregroundStyle(.secondary)
                }
            }
        }
        .navigationDestination(isPresented: $showsProcessing) {
            ProcessingView()
        }
        .onDisappear {
            viewModel.stopPlayback()
        }
    }

    private var previewImageName: String {
        switch viewModel.videoSource {
        case .asset(let name): name
        }
    }

    private var playbackControls: some View {
        HStack(spacing: 12) {
            Button {
                viewModel.togglePlayback()
            } label: {
                Image(systemName: viewModel.isPlaying ? "pause.fill" : "play.fill")
                    .font(.subheadline.weight(.semibold))
                    .frame(width: 36, height: 36)
            }
            .buttonStyle(.plain)
            .background(Color(.secondarySystemBackground), in: Circle())
            .accessibilityLabel(viewModel.isPlaying ? "Pause" : "Play")

            HStack(alignment: .firstTextBaseline, spacing: 5) {
                Text(viewModel.formattedCurrentTime)
                    .font(.subheadline.weight(.semibold))
                    .foregroundStyle(.primary)
                Text("/ \(viewModel.formattedDuration)")
                    .font(.footnote)
                    .foregroundStyle(.secondary)
            }
            .monospacedDigit()

            Spacer()
        }
    }

    private var continueButton: some View {
        VStack(spacing: 0) {
            Divider()
            PrimaryButton(title: "Continue", isEnabled: viewModel.canContinue) {
                guard viewModel.canContinue else { return }
                showsProcessing = true
            }
            .padding(.horizontal, 16)
            .padding(.vertical, 12)
        }
        .background(.bar)
    }
}

